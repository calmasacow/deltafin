"""Decode-path attention/norm fast paths.  Every knob here defaults to the
behaviour the repo shipped with, so the orchestrator can A/B each one alone.

Measured per-op cost of ONE KDA layer at T=1 on this machine (fp32, MPS,
median of 15, each op serialized with torch.mps.synchronize):

    q/k/v_proj      3.17 ms   (3 x 7168x12288 fp32 GEMV = 1.06 GB read)
    g_proj          1.21 ms
    o_proj          1.21 ms
    recurrence      3.12 ms   <- CPU hop; 1.17 ms if left on MPS
    short conv x3   0.63 ms   <- 0.38 ms as a 4-tap gather+mul+sum
    f_a/f_b/b_proj  0.30 ms
    o_norm          0.25 ms
    ------------------------
    KDA forward     9.59 ms   (whole module, one sync)
    MLA forward     3.45 ms
    input/post RMSNorm 0.25 ms each, _apply_attn_res 0.43 ms x2/layer
    MoE gate 0.42 ms, shared experts 1.81 ms, latent down/up 0.83 ms

The five big projections are 6.05 of the 9.59 ms and are already running at
~291 GB/s (M1 Max peak is 400), i.e. they are pure fp32 weight bandwidth and
cannot be cut without changing the numerics.  Everything that is *not*
bandwidth is dispatch overhead and device hops -- that is what this file
attacks.

Flags
-----
K3_KDA_RECUR = mps (default) | cpu
    Where the T<=4 KDA state recurrence runs.  The historical "cpu" path moves
    240 KB of q/k/v/g/beta D2H, runs ~10 tiny ops, and copies the output back.
    Re-measurement found 1.17 ms on MPS against 3.12 ms for the CPU round trip,
    so "mps" is the shipped default.  The CPU hop is also a full GPU barrier 69
    times per token.

K3_SHORTCONV = mulsum (default) | conv1d
    At T=1 the depthwise conv is a per-channel dot of 4 taps over 12,288
    channels; F.conv1d with 12,288 groups is a bad fit for MPS.  "mulsum"
    builds the 4-tap window once (which IS the new conv cache -- bit-identical,
    verified) and does one [D,4]*[D,4] multiply + sum.

K3_COMPILE = 0 (default) | attn | layer      (K3_COMPILE=1 means "attn")
K3_COMPILE_MODE = default | reduce-overhead | max-autotune
    torch.compile the reused template layers.  Template reuse gives stable
    shapes and stable parameter identities, but layer_idx is still rewritten on
    the template every layer; under "attn" the cache slot is lifted out of the
    graph (see _Slot1) so dynamo never sees layer_idx and there are only 2-3
    graphs instead of 93.
"""
import os
import time

import torch

RECUR = os.environ.get("K3_KDA_RECUR", "mps")            # mps | cpu
SHORTCONV = os.environ.get("K3_SHORTCONV", "mulsum")     # mulsum | conv1d
COMPILE = os.environ.get("K3_COMPILE", "0")              # 0 | attn | layer | 1
COMPILE_MODE = os.environ.get("K3_COMPILE_MODE", "default")
if COMPILE == "1":
    COMPILE = "attn"

# The line is useful even with no overrides: it proves that the live lower-level
# shims and this reporting module agree on the optimized defaults.
ACTIVE = True
STATS = {"compile_s": 0.0, "graphs": 0}

_ORIG = {}


def describe():
    s = f"recur={RECUR} shortconv={SHORTCONV} compile={COMPILE}"
    if COMPILE != "0":
        s += f"({COMPILE_MODE})"
        if STATS["graphs"]:
            s += f" graphs={STATS['graphs']} compile_s={STATS['compile_s']:.1f}"
    return s


class _Slot1:
    """Single-slot stand-in for KimiDynamicCache.

    The template layers are reused for all 93 layers with `layer_idx` rewritten
    each time, so any graph that reads `cache.conv_states[self.layer_idx]` gets
    a guard on the *value* of layer_idx and recompiles 93 times.  Reading and
    writing the real cache outside the compiled region -- and pinning
    layer_idx=0 inside it -- keeps it to one graph per template.
    """
    __slots__ = ("conv_states", "recurrent_states", "key_cache", "value_cache")

    def __init__(self, conv=None, rec=None, k=None, v=None):
        self.conv_states = [conv]
        self.recurrent_states = [rec]
        self.key_cache = [k]
        self.value_cache = [v]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.key_cache[0] is None:
            self.key_cache[0] = key_states
            self.value_cache[0] = value_states
        else:
            self.key_cache[0] = torch.cat([self.key_cache[0], key_states], dim=2)
            self.value_cache[0] = torch.cat([self.value_cache[0], value_states], dim=2)
        return self.key_cache[0], self.value_cache[0]


_compiled = {}


def _get_compiled(mod, fn):
    """One compiled artifact per module instance (2-3 templates), built lazily
    so the compile cost is paid on the first token and never again."""
    key = id(mod)
    got = _compiled.get(key)
    if got is None:
        kw = {"dynamic": False}
        if COMPILE_MODE != "default":
            kw["mode"] = COMPILE_MODE
        t0 = time.time()
        got = torch.compile(fn.__get__(mod), **kw)
        STATS["compile_s"] += time.time() - t0     # dynamo compiles lazily; the
        STATS["graphs"] += 1                       # real cost lands on call 1
        _compiled[key] = got
    return got


def install(ml):
    """Apply the enabled patches to Moonshot's modeling module."""
    if COMPILE == "0":
        return
    try:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = max(getattr(dynamo.config,
                                                     "cache_size_limit", 8), 64)
    except Exception:
        pass

    if COMPILE == "layer":
        orig_layer = _ORIG.setdefault("layer", ml.KimiDecoderLayer.forward)

        def layer_forward(self, *a, **kw):
            return _get_compiled(self, orig_layer)(*a, **kw)

        ml.KimiDecoderLayer.forward = layer_forward
        return

    orig_kda = _ORIG.setdefault("kda", ml.KimiDeltaAttention.forward)
    orig_mla = _ORIG.setdefault("mla", ml.KimiMLAAttention.forward)

    def kda_forward(self, hidden_states, attention_mask=None, cache_params=None, **kw):
        # A 2D mask is the varlen/unpad path: data-dependent shapes, never worth
        # compiling. A 4D causal mask is ignored by KDA and passed straight on.
        if cache_params is None or (attention_mask is not None
                                    and attention_mask.dim() == 2):
            return orig_kda(self, hidden_states, attention_mask, cache_params, **kw)
        li = self.layer_idx
        slot = _Slot1(cache_params.conv_states[li], cache_params.recurrent_states[li])
        fn = _get_compiled(self, orig_kda)
        self.layer_idx = 0
        try:
            out = fn(hidden_states, attention_mask, slot, **kw)
        finally:
            self.layer_idx = li
        cache_params.conv_states[li] = slot.conv_states[0]
        cache_params.recurrent_states[li] = slot.recurrent_states[0]
        return out

    def mla_forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_values=None, **kw):
        if past_key_values is None:
            return orig_mla(self, hidden_states, attention_mask, position_ids,
                            past_key_values, **kw)
        li = self.layer_idx
        slot = _Slot1(k=past_key_values.key_cache[li],
                      v=past_key_values.value_cache[li])
        fn = _get_compiled(self, orig_mla)
        self.layer_idx = 0
        try:
            out = fn(hidden_states, attention_mask, position_ids, slot, **kw)
        finally:
            self.layer_idx = li
        past_key_values.key_cache[li] = slot.key_cache[0]
        past_key_values.value_cache[li] = slot.value_cache[0]
        return out

    ml.KimiDeltaAttention.forward = kda_forward
    ml.KimiMLAAttention.forward = mla_forward
