"""Decode-path attention/norm fast paths.

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
K3_KDA_RECUR = device (default) | cpu     ("mps" is a legacy alias)
    Where the T<=4 KDA state recurrence runs.  The historical "cpu" path moves
    240 KB of q/k/v/g/beta D2H, runs ~10 tiny ops, and copies the output back.
    Re-measurement found 1.17 ms on MPS against 3.12 ms for the CPU round trip,
    so "device" is the MPS default. On CUDA and CPU it accurately means the
    selected device; only MPS supports the optional historical CPU hop.

K3_SHORTCONV = auto (default) | mulsum | conv1d
    During decode/speculation the depthwise conv is a per-channel dot of 4 taps
    over 12,288 channels; F.conv1d with 12,288 groups is a poor tiny-width
    kernel shape. "mulsum" builds every sliding 4-tap window once and does one
    [B,D,T,4]*[D,4] multiply + sum. Its cache state is bit-identical and output
    is within the strict numerical gate versus repeated T=1 calls. "auto" uses
    it through T=9 on CPU, where the real shape has a decisive win; accelerator
    auto mode retains the established T=1 path until a backend-specific T>1
    gate passes. K3_SHORTCONV_AUTO_MAX_T overrides that boundary, and either
    implementation remains forceable.

K3_COMPILE = 0 (default) | attn | layer      (K3_COMPILE=1 means "attn")
K3_COMPILE_MODE = default | reduce-overhead | max-autotune
    torch.compile the reused template layers.  Template reuse gives stable
    shapes and stable parameter identities, but layer_idx is still rewritten on
    the template every layer; under "attn" the cache slot is lifted out of the
    graph (see _Slot1) so dynamo never sees layer_idx and there are only 2-3
    graphs instead of 93.

K3_MLA_CPU_SDPA = 0 (default) | 1
    Opt in to zero-padding each fp32 MLA value history from width 128 to 192,
    selecting PyTorch's native CPU FlashAttention operator, and slicing the
    output back to 128. Eligibility is based on runtime operator activation
    and the exact tested batch/head/dtype/T/context/mask contract, never an OS
    or CPU product name. The cache remains 128-wide. Any missing capability,
    unsupported shape, allocation failure, or operator failure uses eager
    attention and is reported by cpu_sdpa_status().
"""
import os
import threading
import time

import torch
import torch.nn.functional as F
import fla.modules as _fla_modules
import fla.ops.kda as _fla_kda

RECUR = _fla_kda.RECUR                         # device | cpu
SHORTCONV = _fla_modules.SHORTCONV             # auto | mulsum | conv1d
COMPILE = os.environ.get("K3_COMPILE", "0")              # 0 | attn | layer | 1
COMPILE_MODE = os.environ.get("K3_COMPILE_MODE", "default")
if COMPILE == "1":
    COMPILE = "attn"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_CPU_SDPA_MODE = os.environ.get("K3_MLA_CPU_SDPA", "0").strip().lower()
if _CPU_SDPA_MODE not in _TRUE | _FALSE:
    raise ValueError("K3_MLA_CPU_SDPA must be 0/off/false or 1/on/true")
CPU_SDPA_REQUESTED = _CPU_SDPA_MODE in _TRUE
_CPU_SDPA_TOKENS = frozenset((1, 2, 8, 9))
_CPU_SDPA_CONTEXT_BUCKETS = (
    (32, 127),
    (128, 511),
    (512, 1023),
    (1024, 1024),
)
_CPU_SDPA_NATIVE_OPERATOR = (
    "aten::_scaled_dot_product_flash_attention_for_cpu"
)

# The line is useful even with no overrides: it proves that the live lower-level
# shims and this reporting module agree on the optimized defaults.
ACTIVE = True
STATS = {"compile_s": 0.0, "graphs": 0}

_ORIG = {}
_CPU_SDPA_LOCK = threading.RLock()
_CPU_SDPA_INSTALLED = False
_CPU_SDPA_INSTALL_REASON = None
_CPU_SDPA_KEYS = {}
_CPU_SDPA_STATS = {
    "calls": 0,
    "fast_calls": 0,
    "eager_bypass_calls": 0,
    "eager_fallback_calls": 0,
    "canary_attempts": 0,
    "canary_passes": 0,
    "canary_failures": 0,
    "runtime_failures": 0,
    "disabled_key_bypasses": 0,
    "transient_padding_bytes": 0,
    "peak_transient_padding_bytes": 0,
}
_CPU_SDPA_BYPASS_REASONS = {}


def describe(device=None):
    """Describe resolved execution modes, including the selected backend."""
    device_type = getattr(device, "type", None)
    recur = (
        f"device({device_type})"
        if RECUR == "device" and device_type is not None else RECUR
    )
    shortconv = SHORTCONV
    if device_type is not None and SHORTCONV == "auto":
        resolved = _fla_modules.resolve_shortconv_mode(device_type, SHORTCONV)
        shortconv = f"auto->{resolved}({device_type})"
    cpu_sdpa = "requested" if CPU_SDPA_REQUESTED else "off"
    if _CPU_SDPA_INSTALLED:
        cpu_sdpa = "guarded"
    elif CPU_SDPA_REQUESTED and _CPU_SDPA_INSTALL_REASON:
        cpu_sdpa = f"disabled({_CPU_SDPA_INSTALL_REASON})"
    s = (
        f"recur={recur} shortconv={shortconv} compile={COMPILE} "
        f"mla_cpu_sdpa={cpu_sdpa}"
    )
    if COMPILE != "0":
        s += f"({COMPILE_MODE})"
        if STATS["graphs"]:
            s += f" graphs={STATS['graphs']} compile_s={STATS['compile_s']:.1f}"
    return s


def _bump_cpu_sdpa(name, amount=1):
    with _CPU_SDPA_LOCK:
        _CPU_SDPA_STATS[name] += amount


def _record_cpu_sdpa_bypass(reason):
    with _CPU_SDPA_LOCK:
        _CPU_SDPA_STATS["eager_bypass_calls"] += 1
        _CPU_SDPA_BYPASS_REASONS[reason] = (
            _CPU_SDPA_BYPASS_REASONS.get(reason, 0) + 1
        )


def _cpu_capability_name():
    getter = getattr(getattr(torch.backends, "cpu", None),
                     "get_cpu_capability", None)
    if getter is None:
        getter = getattr(getattr(torch, "_C", None),
                         "_get_cpu_capability", None)
    if getter is None:
        return "unknown"
    try:
        return str(getter())
    except Exception:
        return "unknown"


def _context_bucket(context):
    for lower, upper in _CPU_SDPA_CONTEXT_BUCKETS:
        if lower <= context <= upper:
            return f"{lower}-{upper}"
    return None


def _mask_contract(mask, query, context):
    tokens = query.shape[-2]
    if mask is None:
        return ("none", None) if tokens == 1 else (
            None, "multi-token call requires an additive causal mask")
    if tokens == 1:
        return None, "single-token call requires the tested no-mask contract"
    if not isinstance(mask, torch.Tensor):
        return None, "attention mask is not a tensor"
    if mask.device.type != "cpu":
        return None, "attention mask is not on CPU"
    if mask.dtype != torch.float32:
        return None, "attention mask is not fp32 additive"
    if mask.dim() != 4:
        return None, "attention mask is not rank four"
    if mask.shape[-2] != tokens or mask.shape[-1] != context:
        return None, "attention mask shape is incompatible"
    return "additive-lower-right", None


def _tensor_layout(tensor):
    if tensor.stride(-1) != 1:
        return None
    if tensor.stride(-2) != tensor.shape[-1]:
        return None
    if tensor.is_contiguous():
        return "contiguous"
    minimum_head_stride = tensor.shape[-2] * tensor.shape[-1]
    if tensor.stride(1) < minimum_head_stride:
        return None
    if tensor.stride(1) % tensor.shape[-1]:
        return None
    return "slab-prefix"


def _cpu_sdpa_key(module, query, key, value, attention_mask, dropout):
    """Return a runtime/capability key or a fail-closed bypass reason."""
    if not CPU_SDPA_REQUESTED:
        return None, "not requested"
    if COMPILE != "0":
        return None, "K3_COMPILE is active"
    if torch.is_grad_enabled() or getattr(module, "training", False):
        return None, "training or grad-enabled call"
    if float(dropout) != 0.0:
        return None, "dropout is nonzero"
    if query.device.type != "cpu":
        return None, "device is not CPU"
    if key.device != query.device or value.device != query.device:
        return None, "q/k/v devices differ"
    if (
        query.dtype != torch.float32
        or key.dtype != torch.float32
        or value.dtype != torch.float32
    ):
        return None, "q/k/v are not fp32"
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return None, "q/k/v are not rank four"
    if (
        query.shape[0] != 1
        or key.shape[0] != 1
        or value.shape[0] != 1
    ):
        return None, "batch size is not one"
    if (
        query.shape[1] != 96
        or key.shape[1] != 96
        or value.shape[1] != 96
    ):
        return None, "head count is not 96"
    if (
        query.shape[-1] != 192
        or key.shape[-1] != 192
        or value.shape[-1] != 128
    ):
        return None, "head widths are not qk=192/value=128"
    if key.shape[-2] != value.shape[-2]:
        return None, "key/value contexts differ"
    if getattr(module, "num_key_value_groups", 1) != 1:
        return None, "grouped key/value heads are unsupported"
    layouts = tuple(_tensor_layout(tensor) for tensor in (
        query, key, value
    ))
    if any(layout is None for layout in layouts):
        return None, "q/k/v layout is outside the tested contract"

    tokens = query.shape[-2]
    context = key.shape[-2]
    if tokens not in _CPU_SDPA_TOKENS:
        return None, f"T={tokens} is outside the tested set"
    bucket = _context_bucket(context)
    if bucket is None:
        return None, f"context={context} is outside the tested range"
    mask_contract, reason = _mask_contract(
        attention_mask, query, context
    )
    if reason is not None:
        return None, reason
    key_label = (
        f"torch={torch.__version__}|cpu={_cpu_capability_name()}|"
        f"threads={torch.get_num_threads()}|"
        f"interop={torch.get_num_interop_threads()}|"
        f"dtype=fp32|B=1|H=96|"
        f"qk=192|v=128|T={tokens}|L={bucket}|mask={mask_contract}"
        f"|layout={'/'.join(layouts)}"
    )
    return key_label, None


def _cpu_sdpa_call(query, key, value, attention_mask, scaling):
    context = key.shape[-2]
    mask = (
        attention_mask[:, :, :query.shape[-2], :context]
        if attention_mask is not None else None
    )
    padded = F.pad(value, (0, query.shape[-1] - value.shape[-1]))
    transient_bytes = padded.numel() * padded.element_size()
    output = F.scaled_dot_product_attention(
        query,
        key,
        padded,
        attn_mask=mask,
        dropout_p=0.0,
        scale=scaling,
    )[..., :value.shape[-1]]
    output = output.transpose(1, 2).contiguous()
    return output, transient_bytes


def _profile_cpu_sdpa(query, key, value, attention_mask, scaling):
    profiler = getattr(torch, "profiler", None)
    if profiler is None:
        raise RuntimeError("torch.profiler is unavailable")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU]
    ) as profile:
        output, transient_bytes = _cpu_sdpa_call(
            query, key, value, attention_mask, scaling
        )
    operators = sorted({event.key for event in profile.key_averages()})
    if _CPU_SDPA_NATIVE_OPERATOR not in operators:
        raise RuntimeError(
            "native CPU FlashAttention operator was not observed; "
            f"operators={operators}"
        )
    return output, transient_bytes, operators


def _disable_cpu_sdpa_key(key, reason, *, canary):
    with _CPU_SDPA_LOCK:
        previous = _CPU_SDPA_KEYS.get(key) or {}
        _CPU_SDPA_KEYS[key] = {
            "enabled": False,
            "reason": reason,
            "operator": previous.get("operator"),
            "failed_after_activation": not canary,
        }
        if canary:
            _CPU_SDPA_STATS["canary_failures"] += 1
        else:
            _CPU_SDPA_STATS["runtime_failures"] += 1


def _make_cpu_sdpa_forward(original):
    def cpu_sdpa_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        _bump_cpu_sdpa("calls")
        try:
            key_label, reason = _cpu_sdpa_key(
                module, query, key, value, attention_mask, dropout
            )
        except Exception as error:
            key_label = None
            reason = (
                "eligibility check failed: "
                f"{type(error).__name__}: {error}"
            )
        if key_label is None:
            _record_cpu_sdpa_bypass(reason)
            return original(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=dropout, **kwargs
            )

        with _CPU_SDPA_LOCK:
            state = _CPU_SDPA_KEYS.get(key_label)
            if state is not None and not state["enabled"]:
                _CPU_SDPA_STATS["disabled_key_bypasses"] += 1
                _CPU_SDPA_STATS["eager_fallback_calls"] += 1
                disabled = True
            else:
                disabled = False
        if disabled:
            return original(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=dropout, **kwargs
            )

        if state is None:
            with _CPU_SDPA_LOCK:
                # Serialize the first real call for a key so concurrent
                # requests cannot race two process-wide profilers.
                state = _CPU_SDPA_KEYS.get(key_label)
                if state is None:
                    _CPU_SDPA_STATS["canary_attempts"] += 1
                    try:
                        output, transient_bytes, _ = _profile_cpu_sdpa(
                            query, key, value, attention_mask, scaling
                        )
                    except Exception as error:
                        failure = f"{type(error).__name__}: {error}"
                        _CPU_SDPA_KEYS[key_label] = {
                            "enabled": False,
                            "reason": failure,
                            "operator": None,
                            "failed_after_activation": False,
                        }
                        _CPU_SDPA_STATS["canary_failures"] += 1
                    else:
                        _CPU_SDPA_KEYS[key_label] = {
                            "enabled": True,
                            "reason": None,
                            "operator": _CPU_SDPA_NATIVE_OPERATOR,
                            "failed_after_activation": False,
                        }
                        _CPU_SDPA_STATS["canary_passes"] += 1
                        _CPU_SDPA_STATS["fast_calls"] += 1
                        _CPU_SDPA_STATS[
                            "transient_padding_bytes"
                        ] += transient_bytes
                        _CPU_SDPA_STATS[
                            "peak_transient_padding_bytes"
                        ] = max(
                            _CPU_SDPA_STATS[
                                "peak_transient_padding_bytes"
                            ],
                            transient_bytes,
                        )
                        return output, None
                    state = _CPU_SDPA_KEYS[key_label]
                if not state["enabled"]:
                    _CPU_SDPA_STATS["eager_fallback_calls"] += 1
                    return original(
                        module, query, key, value, attention_mask,
                        scaling=scaling, dropout=dropout, **kwargs
                    )

        try:
            output, transient_bytes = _cpu_sdpa_call(
                query, key, value, attention_mask, scaling
            )
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            _disable_cpu_sdpa_key(
                key_label, failure, canary=False
            )
            _bump_cpu_sdpa("eager_fallback_calls")
            return original(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=dropout, **kwargs
            )
        with _CPU_SDPA_LOCK:
            _CPU_SDPA_STATS["fast_calls"] += 1
            _CPU_SDPA_STATS[
                "transient_padding_bytes"
            ] += transient_bytes
            _CPU_SDPA_STATS[
                "peak_transient_padding_bytes"
            ] = max(
                _CPU_SDPA_STATS["peak_transient_padding_bytes"],
                transient_bytes,
            )
        return output, None

    return cpu_sdpa_forward


def cpu_sdpa_status():
    """Serializable fail-closed telemetry for the opt-in MLA CPU path."""
    with _CPU_SDPA_LOCK:
        return {
            "requested": CPU_SDPA_REQUESTED,
            "installed": _CPU_SDPA_INSTALLED,
            "install_reason": _CPU_SDPA_INSTALL_REASON,
            "native_operator": _CPU_SDPA_NATIVE_OPERATOR,
            "runtime": {
                "torch": torch.__version__,
                "cpu_capability": _cpu_capability_name(),
                "threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
            },
            "tested_contract": {
                "device": "cpu",
                "dtype": "torch.float32",
                "batch": 1,
                "heads": 96,
                "qk_head_dim": 192,
                "value_head_dim": 128,
                "tokens": sorted(_CPU_SDPA_TOKENS),
                "context_buckets": [
                    [lower, upper]
                    for lower, upper in _CPU_SDPA_CONTEXT_BUCKETS
                ],
                "persistent_padded_kv": False,
                "persistent_padding_bytes": 0,
            },
            **_CPU_SDPA_STATS,
            "bypass_reasons": dict(_CPU_SDPA_BYPASS_REASONS),
            "keys": [
                {"key": key, **value}
                for key, value in sorted(_CPU_SDPA_KEYS.items())
            ],
        }


def _reset_cpu_sdpa_for_tests():
    """Reset process telemetry/canaries; never called by the runtime."""
    with _CPU_SDPA_LOCK:
        _CPU_SDPA_KEYS.clear()
        _CPU_SDPA_BYPASS_REASONS.clear()
        for name in _CPU_SDPA_STATS:
            _CPU_SDPA_STATS[name] = 0


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


def _portable_cache_reorder(self, beam_idx):
    """Reorder every cache tensor with indices resident on its own device."""
    indices_by_device = {}
    has_slab_bookkeeping = all(hasattr(self, name) for name in (
        "_key_storage", "_value_storage", "_kv_capacity"
    ))

    def select_rows(tensor):
        device = tensor.device
        indices = indices_by_device.get(device)
        if indices is None:
            indices = beam_idx.to(device=device, dtype=torch.long)
            indices_by_device[device] = indices
        return tensor.index_select(0, indices)

    for layer_idx in range(len(self.key_cache)):
        if self.key_cache[layer_idx] is not None:
            self.key_cache[layer_idx] = select_rows(
                self.key_cache[layer_idx])
            self.value_cache[layer_idx] = select_rows(
                self.value_cache[layer_idx])
            if has_slab_bookkeeping:
                self._key_storage[layer_idx] = self.key_cache[layer_idx]
                self._value_storage[layer_idx] = self.value_cache[layer_idx]
                self._kv_capacity[layer_idx] = (
                    self.key_cache[layer_idx].shape[2])

        if self.conv_states[layer_idx] is not None:
            q_conv, k_conv, v_conv = self.conv_states[layer_idx]
            self.conv_states[layer_idx] = (
                select_rows(q_conv),
                select_rows(k_conv),
                select_rows(v_conv),
            )
        if self.recurrent_states[layer_idx] is not None:
            self.recurrent_states[layer_idx] = select_rows(
                self.recurrent_states[layer_idx])


_portable_cache_reorder._k3_device_local_indices = True


def install_cache_portability(ml):
    """Install exact cache fixes that apply regardless of compile settings."""
    cache_class = getattr(ml, "KimiDynamicCache", None)
    if cache_class is None:
        return False
    current = getattr(cache_class, "reorder_cache", None)
    if current is None:
        return False
    if getattr(current, "_k3_device_local_indices", False):
        return True
    _ORIG.setdefault(("cache_reorder", id(cache_class)), current)
    cache_class.reorder_cache = _portable_cache_reorder
    return True


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
    global _CPU_SDPA_INSTALLED, _CPU_SDPA_INSTALL_REASON
    install_cache_portability(ml)
    if CPU_SDPA_REQUESTED:
        if COMPILE != "0":
            _CPU_SDPA_INSTALL_REASON = (
                "K3_COMPILE must be 0 for the guarded SDPA path"
            )
        elif _CPU_SDPA_INSTALLED:
            pass
        else:
            original = _ORIG.setdefault(
                "cpu_sdpa_attention", ml.eager_attention_forward
            )
            ml.eager_attention_forward = _make_cpu_sdpa_forward(original)
            _CPU_SDPA_INSTALLED = True
            _CPU_SDPA_INSTALL_REASON = None
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
