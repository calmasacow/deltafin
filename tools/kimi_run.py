#!/usr/bin/env python3
"""lazy-K3: real Kimi-K3 inference on a 64GB M1 Max by layer-streaming.

Uses Moonshot's own modeling_kimi_linear.py (audited) with a pure-PyTorch fla shim.
Per forward pass, each of the 93 decoder layers is materialized from the local
resident-spine download, routed experts are fetched on demand (HTTP Range, disk
cached) and dequantized from MXFP4, the layer runs in fp32 on CPU, then its
weights are freed. Router selections are logged to router_trace.jsonl.
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))    # fla shim, k3loader, mxfp4
# modeling files imported via tools/k3pkg package

import numpy as np
import torch
import torch.nn as nn

torch.set_grad_enabled(False)
torch.set_num_threads(8)

import k3loader  # noqa: E402
import importlib  # noqa: E402

from k3pkg import modeling_kimi_linear as ml

CFG_JSON = json.load(open(os.path.join(ROOT, "k3-meta/config.json")))["text_config"]
Cfg = getattr(ml, "KimiLinearConfig", None)
if Cfg is None:
    Cfg = importlib.import_module("k3pkg.configuration_kimi_k3").KimiLinearConfig
config = Cfg(**CFG_JSON)
config._attn_implementation = "eager"
H = config.hidden_size
NL = config.num_hidden_layers
PFX = "language_model.model."
DEV = torch.device(os.environ.get("K3_DEV", "cpu"))        # cpu | mps
SPINE = os.environ.get("K3_SPINE", "bf16")                 # bf16 | int8
# K3_APPROX=1 = "approx mode": approximate numerics (fp16 weights) + n-gram
# speculation. Output stays coherent but near-tie tokens may differ from the
# fp32 reference — never use for oracle runs. Speed effect is unproven until a
# quiet-machine A/B; if it measures faster it can earn a faster name.
APPROX = os.environ.get("K3_APPROX", "0") == "1"
DT = torch.float16 if (APPROX or os.environ.get("K3_DTYPE", "fp32") == "fp16") else torch.float32
INT8_DIR = os.path.join(ROOT, "k3-resident-int8/tensors")
TRACE = open(os.path.join(ROOT, "k3-meta/router_trace.jsonl"), "a")
TIMES = {"resident_io": 0.0, "expert_fetch": 0.0, "compute": 0.0, "moe_kernel": 0.0}
PROFILE = os.environ.get("K3_PROFILE", "0") == "1"
PROF = {"kda": 0.0, "mla": 0.0, "n_kda": 0, "n_mla": 0}


def set_param(root, dotted, tensor):
    obj = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))


def _load_int8(full):
    """Return dequantized fp32 tensor on DEV from the int8 spine, or None."""
    op = os.path.join(INT8_DIR, full + ".i8")
    if not os.path.exists(op):
        return None
    shape = k3loader.INV[full]["shape"]
    q = torch.frombuffer(bytearray(open(op, "rb").read()), dtype=torch.int8).reshape(shape)
    sc = torch.frombuffer(bytearray(open(os.path.join(INT8_DIR, full + ".sc"), "rb").read()),
                          dtype=torch.float16).reshape(shape[0], 1)
    return (q.to(DEV).to(torch.float32) * sc.to(DEV).to(torch.float32)).to(DT)


def materialize_resident(module, prefix):
    t0 = time.time()
    missing = []
    for name, p in list(module.named_parameters()):
        if ".experts." in name:
            continue  # routed experts stay meta until selected
        full = prefix + name
        t = _load_int8(full) if SPINE == "int8" else None
        if t is None:
            try:
                t = k3loader.load_resident(full).to(DEV, DT)
            except KeyError:
                missing.append(full)
                continue
        set_param(module, name, t)
    TIMES["resident_io"] += time.time() - t0
    if missing:
        raise RuntimeError(f"missing resident tensors: {missing[:5]}")


def dematerialize(module):
    for name, p in list(module.named_parameters()):
        if p.device.type != "meta":
            set_param(module, name, torch.empty_like(p, device="meta"))


# --- double-buffered layer loading: a worker thread reads layer N+1's blobs
# (file I/O releases the GIL) while layer N computes; main thread does the
# tensor creation / dequant / device transfer ---------------------------------
import concurrent.futures as _cf  # noqa: E402
_PRELOADER = _cf.ThreadPoolExecutor(1)
PRELOAD = os.environ.get("K3_PRELOAD", "1") == "1"


# --- adaptive RAM budget: pin as many resident layers as this machine affords --
# budget = total RAM - OS/apps reserve (max(10GB, 18%)); ~40% of the spendable
# budget goes to pinned layers (loaded once, never freed); the rest is left to
# the page cache, which holds hot expert .bin files and self-scales with RAM.
# Template-layer buffer reuse: all KDA layers share one shape class, all MLA
# layers another. Two persistent materialized templates + copy_() per layer
# kills the MPS alloc/free churn measured at 1317 -> 288 ms/layer.
TEMPLATES = os.environ.get("K3_TEMPLATES", "1") == "1"


def _ram_budget_layers():
    if TEMPLATES:
        return 0  # templates and per-layer pinning are mutually exclusive
    if os.environ.get("K3_PIN_LAYERS") is not None:
        return int(os.environ["K3_PIN_LAYERS"])
    import subprocess
    total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
    reserve = max(10.0, 0.18 * total_gb)
    budget = float(os.environ.get("K3_RAM_GB", 0)) or (total_gb - reserve)
    overhead = 8.0 + (4.7 if DT == torch.float32 else 2.35) + 2.0   # process+lm_head+transients
    per_layer = (113.5 / NL) * (2 if DT == torch.float32 else 1)    # fp32=2x int8 bytes, fp16=1x
    n = max(0, int(0.4 * (budget - overhead) / per_layer))
    print(f"[ram] total {total_gb:.0f} GB, budget {budget:.1f} GB -> pinning "
          f"{min(n, NL)} of {NL} layers ({min(n, NL) * per_layer:.1f} GB at {DT})", flush=True)
    return min(n, NL)


PIN_N = _ram_budget_layers()


def _read_resident_bytes(module, prefix):
    out = {}
    for name, _ in module.named_parameters():
        if ".experts." in name:
            continue
        full = prefix + name
        if SPINE == "int8":
            op = os.path.join(INT8_DIR, full + ".i8")
            if os.path.exists(op):
                out[full] = ("i8", open(op, "rb").read(),
                             open(os.path.join(INT8_DIR, full + ".sc"), "rb").read())
                continue
        path = os.path.join(k3loader.RES, full)
        if os.path.exists(path):
            out[full] = ("bf16", open(path, "rb").read())
    return out


def _apply_resident(module, prefix, blobs):
    t0 = time.time()
    for name, p in list(module.named_parameters()):
        if ".experts." in name:
            continue
        full = prefix + name
        rec = blobs.get(full)
        if rec is None:
            t = k3loader.load_resident(full).to(DEV, DT)
        elif rec[0] == "i8":
            shape = k3loader.INV[full]["shape"]
            q = torch.frombuffer(bytearray(rec[1]), dtype=torch.int8).reshape(shape)
            sc = torch.frombuffer(bytearray(rec[2]), dtype=torch.float16).reshape(shape[0], 1)
            t = (q.to(DEV).to(torch.float32) * sc.to(DEV).to(torch.float32)).to(DT)
        else:
            meta = k3loader.INV[full]
            t = torch.frombuffer(bytearray(rec[1]),
                                 dtype=k3loader._DT[meta["dtype"]]).reshape(meta["shape"]).to(DEV, DT)
        set_param(module, name, t)
    TIMES["resident_io"] += time.time() - t0


# --- MoE expert lazy materialization + router trace ---------------------------
_step_ctx = {"layer": -1, "step": -1}
_orig_moe_infer = ml.KimiSparseMoeBlock.moe_infer


FAST_MOE = os.environ.get("K3_FAST_MOE", "1") == "1"
import fast_moe  # noqa: E402

if os.environ.get("K3_FETCH", "v2") == "v2":
    import fetch_v2
    k3loader.fetch_experts = fetch_v2.fetch_experts  # 6.4x: coalesced + keep-alive


_LAST_SEL = {}   # layer -> ids selected for the most recent token (prefetch oracle)


def prefetch_prev_token():
    """Fire-and-forget: fetch the previous token's full per-layer expert sets
    (39.7% measured next-token recall); misses stream to disk while layers compute."""
    import threading
    snap = dict(_LAST_SEL)

    def run():
        for li in sorted(snap):
            try:
                k3loader.fetch_experts(li, snap[li], dequant=False)
            except Exception:
                pass
    threading.Thread(target=run, daemon=True).start()


def moe_infer_lazy(self, x, topk_ids, topk_weight):
    li = _step_ctx["layer"]
    ids = sorted(set(topk_ids.view(-1).tolist()))
    _LAST_SEL[li] = ids
    t0 = time.time()
    raw = k3loader.fetch_experts(li, ids, dequant=not FAST_MOE)
    TIMES["expert_fetch"] += time.time() - t0
    TRACE.write(json.dumps({"step": _step_ctx["step"], "layer": li,
                            "ids": topk_ids.view(-1).tolist(),
                            "w": [round(x, 5) for x in topk_weight.view(-1).tolist()]}) + "\n")
    TRACE.flush()
    if FAST_MOE:
        tk = time.time()
        out = fast_moe.moe_infer_fast(x, topk_ids, topk_weight, raw)
        TIMES["moe_kernel"] += time.time() - tk
        return out
    for e, w in raw.items():
        ex = self.experts[e]
        for wn in ("w1", "w2", "w3"):
            set_param(ex, wn + ".weight", w[wn])
    out = _orig_moe_infer(self, x, topk_ids, topk_weight)
    for e in ids:  # free expert weights again
        for wn in ("w1", "w2", "w3"):
            set_param(self.experts[e], wn + ".weight",
                      torch.empty(0, device="meta"))
    return out


ml.KimiSparseMoeBlock.moe_infer = moe_infer_lazy


# --- embeddings via memmap (row reads only) -----------------------------------
class LazyEmbed:
    """bf16 embedding rows from the local blob when present, else per-row HTTP Range."""
    NAME = PFX + "embed_tokens.weight"

    def __init__(self):
        self.path = os.path.join(ROOT, "k3-resident/tensors", self.NAME)
        self.meta = k3loader.INV[self.NAME]
        self.rowbytes = H * 2

    def _row(self, tid):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                f.seek(tid * self.rowbytes)
                return f.read(self.rowbytes)
        m = self.meta
        start = 8 + m["hlen"] + m["offsets"][0] + tid * self.rowbytes
        import urllib.request
        req = urllib.request.Request(
            k3loader.BASE + m["shard"],
            headers={"Range": f"bytes={start}-{start+self.rowbytes-1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def __call__(self, ids):
        buf = b"".join(self._row(int(t)) for t in ids)
        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(len(ids), H)
        return t.to(DEV, DT).unsqueeze(0)  # [1, T, H]


def build_layers():
    layers = []
    if not TEMPLATES:
        with torch.device("meta"):
            for i in range(NL):
                layers.append(ml.KimiDecoderLayer(config, i).eval())
        return layers
    with torch.device("meta"):
        l0 = ml.KimiDecoderLayer(config, 0).eval()        # dense KDA (unique shape)
        tpl_kda = ml.KimiDecoderLayer(config, 1).eval()   # KDA + MoE class
        tpl_mla = ml.KimiDecoderLayer(config, 3).eval()   # MLA + MoE class
    for i in range(NL):
        layers.append(l0 if i == 0 else
                      tpl_kda if config.is_kda_layer(i) else tpl_mla)
    return layers


def copy_resident(module, prefix, blobs):
    """Like _apply_resident but copies into the module's EXISTING buffers
    (first touch still allocates; A_log's checkpoint shape [128] replaces the
    constructor's [96] once, then copies match)."""
    t0 = time.time()
    for name, p in list(module.named_parameters()):
        if ".experts." in name:
            continue
        full = prefix + name
        rec = blobs.get(full)
        if rec is None:
            t = k3loader.load_resident(full).to(DEV, DT)
        elif rec[0] == "i8":
            shape = k3loader.INV[full]["shape"]
            q = torch.frombuffer(bytearray(rec[1]), dtype=torch.int8).reshape(shape)
            sc = torch.frombuffer(bytearray(rec[2]), dtype=torch.float16).reshape(shape[0], 1)
            t = (q.to(DEV).to(torch.float32) * sc.to(DEV).to(torch.float32)).to(DT)
        else:
            meta = k3loader.INV[full]
            t = torch.frombuffer(bytearray(rec[1]),
                                 dtype=k3loader._DT[meta["dtype"]]).reshape(meta["shape"]).to(DEV, DT)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
    TIMES["resident_io"] += time.time() - t0


def causal_mask(T, dtype=None):
    dtype = dtype or DT
    m = torch.zeros(1, 1, T, T, dtype=dtype, device=DEV)
    m.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool, device=DEV), 1),
                   torch.finfo(dtype).min)
    return m


def forward_pass(layers, cache, hidden, step, verbose=True):
    """hidden: [1, T, H] fp32. Returns logits [1, T, vocab]."""
    T = hidden.shape[1]
    mask = causal_mask(T + (cache.get_seq_length() or 0))[:, :, -T:, :] if T > 1 else None
    block_residual = hidden.new_zeros(T, 0, H)

    def _next_unpinned(j):
        while j < NL and j < PIN_N and getattr(layers[j], "_k3_res", False):
            j += 1
        return j

    nxt = _next_unpinned(0)
    fut = (_PRELOADER.submit(_read_resident_bytes, layers[nxt], f"{PFX}layers.{nxt}.")
           if PRELOAD and nxt < NL else None)
    for i, layer in enumerate(layers):
        _step_ctx["layer"] = i
        if TEMPLATES:
            layer.layer_idx = i
            layer.self_attn.layer_idx = i
        pinned = i < PIN_N and getattr(layer, "_k3_res", False)
        if not pinned:
            if PRELOAD and fut is not None and i == nxt:
                blobs = fut.result()
                j = _next_unpinned(i + 1)
                fut = (_PRELOADER.submit(_read_resident_bytes, layers[j], f"{PFX}layers.{j}.")
                       if j < NL else None)
                nxt = j
                (copy_resident if TEMPLATES else _apply_resident)(layer, f"{PFX}layers.{i}.", blobs)
            else:
                if TEMPLATES:
                    copy_resident(layer, f"{PFX}layers.{i}.", _read_resident_bytes(layer, f"{PFX}layers.{i}."))
                else:
                    materialize_resident(layer, f"{PFX}layers.{i}.")
            if i < PIN_N:
                layer._k3_res = True   # pinned from now on
        if PROFILE and DEV.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        hidden, block_residual = layer(
            hidden, attention_mask=mask, position_ids=None,
            past_key_values=cache, use_cache=True, block_residual=block_residual)
        if PROFILE and DEV.type == "mps":
            torch.mps.synchronize()
        dt_layer = time.time() - t0
        TIMES["compute"] += dt_layer
        if PROFILE:
            k = "kda" if layer.is_linear_attn else "mla"
            PROF[k] += dt_layer
            PROF["n_" + k] += 1
        if not TEMPLATES and not (i < PIN_N):
            dematerialize(layer)
        if verbose and (i % 10 == 0 or i == NL - 1):
            print(f"    layer {i:2d}/92 done  (res_io {TIMES['resident_io']:.0f}s "
                  f"exp {TIMES['expert_fetch']:.0f}s comp {TIMES['compute']:.0f}s)",
                  flush=True)
    if PROFILE:
        mk = TIMES["moe_kernel"]
        print(f"[prof] KDA {PROF['kda']:.1f}s/{PROF['n_kda']} MLA {PROF['mla']:.1f}s/{PROF['n_mla']} "
              f"| moe_kernel {mk:.1f}s | fetch {TIMES['expert_fetch']:.1f}s "
              f"| apply {TIMES['resident_io']:.1f}s", flush=True)
    # tail: output attn-res -> final norm -> lm_head
    tail = nn.Module()
    with torch.device("meta"):
        tail.output_attn_res_norm = ml.KimiRMSNorm(H, eps=config.rms_norm_eps)
        tail.output_attn_res_proj = nn.Linear(H, 1, bias=False)
        tail.norm = ml.KimiRMSNorm(H, eps=config.rms_norm_eps)
    materialize_resident(tail, PFX)
    apply_res = getattr(ml, "_apply_attn_res", None) or ml.KimiDecoderLayer._apply_attn_res
    flat = apply_res(hidden.view(-1, H), block_residual,
                     tail.output_attn_res_proj, tail.output_attn_res_norm)
    hidden = tail.norm(flat.view(1, T, H))
    t0 = time.time()
    global _LM_W
    if _LM_W is None:  # resident across tokens: 2.35-4.7GB on DEV, loaded once
        _LM_W = _load_int8("language_model.lm_head.weight") if SPINE == "int8" else None
        if _LM_W is None:
            _LM_W = k3loader.load_resident("language_model.lm_head.weight").to(DEV, DT)
    TIMES["resident_io"] += time.time() - t0
    logits = hidden @ _LM_W.T
    dematerialize(tail)
    return logits


_LM_W = None


# --- n-gram speculative 2-token decode ----------------------------------------
# Resident I/O + compute dominate a token and amortize across batch positions,
# so an accepted free n-gram draft yields 2 tokens for ~1.2x one pass.
def ngram_draft(ids, max_n=6, min_n=2):
    for n in range(min(max_n, len(ids) - 1), min_n - 1, -1):
        suf = ids[-n:]
        for j in range(len(ids) - n - 1, -1, -1):
            if ids[j:j + n] == suf:
                return ids[j + n]
    return None


def snapshot_states(cache):
    snap = {"rec": {}, "conv": {}, "mla": {}}
    for i in range(NL):
        if cache.recurrent_states[i] is not None:
            snap["rec"][i] = cache.recurrent_states[i].clone()
        if cache.conv_states[i] is not None:
            snap["conv"][i] = tuple(c.clone() for c in cache.conv_states[i])
        if cache.key_cache[i] is not None:
            snap["mla"][i] = cache.key_cache[i].shape[2]
    return snap


def restore_states(cache, snap):
    for i, t in snap["rec"].items():
        cache.recurrent_states[i] = t
    for i, c in snap["conv"].items():
        cache.conv_states[i] = c
    for i, L in snap["mla"].items():
        cache.key_cache[i] = cache.key_cache[i][:, :, :L].contiguous()
        cache.value_cache[i] = cache.value_cache[i][:, :, :L].contiguous()


EOS_ID = 163586  # <|end_of_msg|> — K3's generation stop token


def generate(layers, cache, embed, ids, max_new, spec=None, on_token=None,
             verbose_prefill=False, log=lambda *a: None):
    """Greedy generation (+ certified-lossless n-gram speculation).

    Shared by the CLI and the OpenAI-compatible server. Calls on_token(token_id)
    as each token is emitted. Returns the emitted token list; a speculative
    accept may emit one token past EOS_ID — callers trim at EOS_ID."""
    if spec is None:
        spec = os.environ.get("K3_SPEC", "1") == "1"
    generated = []

    def emit(t):
        generated.append(t)
        if on_token:
            on_token(t)

    _step_ctx["step"] = 0
    logits = forward_pass(layers, cache, embed(ids), step=0, verbose=verbose_prefill)
    emit(int(logits[0, -1].argmax()))
    s = 1
    while len(generated) < max_new:
        _step_ctx["step"] = s
        if os.environ.get("K3_PREFETCH", "1") == "1":
            prefetch_prev_token()
        t0 = time.time()
        draft = ngram_draft(ids + generated) if spec else None
        tag = ""
        if draft is not None:
            snap = snapshot_states(cache)
            logits = forward_pass(layers, cache, embed([generated[-1], draft]),
                                  step=s, verbose=False)
            n1 = int(logits[0, 0].argmax())
            if n1 == draft:
                emit(n1)
                emit(int(logits[0, 1].argmax()))
                tag = " spec+2"
            else:
                restore_states(cache, snap)
                logits = forward_pass(layers, cache, embed([generated[-1]]),
                                      step=s, verbose=False)
                emit(int(logits[0, -1].argmax()))
                tag = " spec-miss"
        else:
            logits = forward_pass(layers, cache, embed([generated[-1]]), step=s, verbose=False)
            emit(int(logits[0, -1].argmax()))
        log(s, tag, t0, list(generated))
        s += 1
        if EOS_ID in generated[-2:]:
            break
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--chat", action="store_true", help="use the K3 chat template")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "k3-meta"), trust_remote_code=True)
    if args.chat:
        ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                      tokenize=True, add_generation_prompt=True)
    else:
        ids = tok.encode(args.prompt)
    print(f"prompt tokens ({len(ids)}): {ids}", flush=True)

    layers = build_layers()
    cache = ml.KimiDynamicCache(config)
    embed = LazyEmbed()
    t_start = time.time()
    print(f"=== prefill: {len(ids)} tokens through 93 layers ===", flush=True)
    state = {"first": True}

    def on_token(t):
        if state["first"]:
            state["first"] = False
            print(f"[prefill done in {time.time()-t_start:.0f}s] "
                  f"first token: {t!r} = {tok.decode([t])!r}", flush=True)

    def log(s, tag, t0, gen):
        print(f"[token {s}: {time.time()-t0:.0f}s{tag}] {tok.decode(gen)!r}", flush=True)
        print("   ", k3loader.cache_report(), flush=True)

    generated = generate(layers, cache, embed, ids, args.max_new,
                         on_token=on_token, verbose_prefill=True, log=log)

    print("\n=== RESULT ===")
    print("completion:", tok.decode(generated))
    print(f"total {time.time()-t_start:.0f}s | times {TIMES}")
    print(k3loader.cache_report())
    TRACE.close()


if __name__ == "__main__":
    main()
