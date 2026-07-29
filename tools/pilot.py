"""PILOT — router-lookahead expert prefetch for lazy-K3.

Every layer today stalls: layer L's router names 16 experts, we read 281 MB, then
we compute.  The reads and the compute are serialised when they could overlap.

The trick (colibri, research/study/colibriCore.md section 3, colibri.c:3847-3909):
after layer L's ATTENTION but before layer L's MoE, take layer L's *pre-MoE*
hidden state, re-normalise it with layer L+1's post-attention RMSNorm, run layer
L+1's ROUTER on it, take the top-k and start those expert reads in the
background.  The state is not yet exact for L+1 (neither L's MoE output nor
L+1's attention delta has been added), which is exactly why it is a prediction.
colibri measured 71.6% top-k recall one layer ahead this way, versus 41.3% for
previous-token identity; our own previous-token recall measured 30.8%, which is
why K3_EXPERT_PREFETCH never paid.

Nothing here can change the model's output.  The real router at layer L+1 still
decides what is computed; a misprediction costs a wasted 17.5 MB read and
nothing else.

Env flags (all default to current behaviour):
  K3_PILOT=1              enable router lookahead + background reads
  K3_PILOT_TWO=1          colibri's refinement: add layer L's *shared* expert
                          output to the hidden state before running L+1's router
                          (colibri measured +2.3% recall; K3 has 2 shared experts)
  K3_PILOT_K=<n>          how many experts to predict (default = router top_k=16)
  K3_PILOT_MAX_PREFETCH=n cap on the per-layer prefetch union during prefill,
                          where the union over T positions can reach T*16 (32)
  K3_PILOT_GATE_DT=       resident gate-weight cache dtype: int8 (default,
                          ~0.56 GB) | fp32 (2.4 GB) | fp16 | bf16 (1.2 GB).
                          int8 uses the checkpoint's existing row scales and
                          torch._weight_int8pack_mm without dequantizing.
  K3_PILOT_MEASURE_ONLY=1 colibri's LOOKA: compute recall, never issue a read
  K3_PILOT_SELFTEST=1     also route layer L with the cached layer-L gate and
                          report recall against the model's own routing; this is
                          a check on the gate cache + router mirror, so it must
                          print 100.0%
  K3_PILOT_ASYNC_DRAIN=1  hand mispredicted in-flight reads to a background
                          thread instead of blocking the demand path on them
  K3_PILOT_NPZ=0          disable the .npz half of the prefetch (see below)
  K3_PILOT_NPZ_WORKERS=n  threads for it (4)
  K3_PILOT_BIN=0          disable the .bin half.  Worth benchmarking: with the
                          expert cache in its current state the reference prompt
                          routes to .bin experts 1 time in 5742, so every .bin
                          speculation is wasted bandwidth (measured 472 wasted
                          reads = 8.3 GB in one prefill, 0 hits)
  K3_PROFILE=1            print the per-layer recall table at the end of a pass

Two read paths, and this matters more than anything else here.  fetch_v2's pread
pool only handles experts cached as `L<l>-E<e>.bin`; experts still in the older
`.npz` form go down a separate np.load path (_PreadReader.read_layer's `other`
list) that fetch_v2.prefetch_layer() cannot touch.  On this machine 12,668 of the
82,432 experts are .npz-only, and they are exactly the hot ones — for the
reference prompt 5,741 of 5,742 routed experts are .npz.  So a .bin-only
prefetch prefetches nothing that is ever used, which is a sufficient explanation
for K3_EXPERT_PREFETCH's measured zero gain all by itself, independent of recall.
PILOT therefore issues predictions down BOTH paths: .bin experts through
fetch_v2's pread pool as before, .npz experts through a small pool of our own
that memoises fetch_v2._cache_load, so a confirmed prediction is a wait on an
already-running load rather than a second read.

The gate cache: layer L+1's gate.weight must be resident when layer L runs, but
the driver copies one layer at a time into shared templates.  So all 92 routers
(gate.weight [896,7168], e_score_correction_bias [896], and the
post_attention_layernorm weight [7168] needed to re-normalise) are loaded once at
startup through the *same* loader the templates use — int8 spine when
K3_SPINE=int8 — so the cached router is bit-identical to the one the model runs.
"""
import concurrent.futures
import os
import threading
import time

import torch

PILOT = os.environ.get("K3_PILOT", "1") == "1"
NPZ = os.environ.get("K3_PILOT_NPZ", "1") == "1"
NPZ_WORKERS = int(os.environ.get("K3_PILOT_NPZ_WORKERS", "4"))
BIN = os.environ.get("K3_PILOT_BIN", "1") == "1"
PILOT_TWO = os.environ.get("K3_PILOT_TWO", "0") == "1"
MEASURE_ONLY = os.environ.get("K3_PILOT_MEASURE_ONLY", "0") == "1"
SELFTEST = os.environ.get("K3_PILOT_SELFTEST", "0") == "1"
ASYNC_DRAIN = os.environ.get("K3_PILOT_ASYNC_DRAIN", "0") == "1"
GATE_DT = os.environ.get("K3_PILOT_GATE_DT", "int8")
_K_ENV = int(os.environ.get("K3_PILOT_K", "0"))
MAX_PREFETCH = int(os.environ.get("K3_PILOT_MAX_PREFETCH", "32"))

_DTS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# --- state -------------------------------------------------------------------
_W = {}          # layer -> gate.weight            [num_experts, H]
_S = {}          # layer -> row scales (native-int8 gates only)
_B = {}          # layer -> e_score_correction_bias [num_experts]
_LN = {}         # layer -> post_attention_layernorm.weight [H]
_CFG = {}        # top_k / eps / NL / first_moe
_READY = False
_BROKEN = False
_INT8_WARNED = False

_CAP = {"x": None}      # pre-MoE hidden captured from post_attention_layernorm
_PRED = {}              # layer -> [[eid, ...], ...] per token position
_PREF = {}              # layer -> union of predicted eids, not yet issued
_SELF = {}              # layer -> self-test routing rows (SELFTEST only)

# recall counters: layer -> [matched, total]
_REC = {}         # every pass
_REC_DEC = {}     # single-position passes only, i.e. decode — the number that
                  # matters, and the one comparable to colibri's 71.6%
_REC_SELF = {}
STATS = {"predict_s": 0.0, "issued_layers": 0, "issued_bin": 0, "issued_npz": 0,
         "gate_bytes": 0, "build_s": 0.0, "npz_hits": 0, "npz_wasted": 0}


def enabled():
    return PILOT and not _BROKEN


# --- gate cache --------------------------------------------------------------
def init(
        config, dev, load, prefix="language_model.model.", load_packed=None,
        native_int8=None):
    """Load every MoE layer's router once.  `load(name)` must be the driver's own
    resident loader so the cached gate matches the one the model will run."""
    global _READY, _BROKEN
    if _READY or _BROKEN:
        return
    t0 = time.time()
    nl = config.num_hidden_layers
    first = getattr(config, "first_k_dense_replace", 1)
    freq = getattr(config, "moe_layer_freq", 1)
    effective_dt = GATE_DT
    if GATE_DT == "int8":
        if native_int8 is None:
            probe = getattr(
                getattr(torch, "_C", None),
                "_dispatch_has_kernel_for_dispatch_key",
                None,
            )
            try:
                native_int8 = (
                    dev.type == "mps"
                    and hasattr(torch, "_weight_int8pack_mm")
                    and (probe is None or bool(probe(
                        "aten::_weight_int8pack_mm", "MPS")))
                )
            except Exception:
                native_int8 = False
        if not native_int8 or load_packed is None:
            effective_dt = "fp32"
            print("[pilot] native-int8 gate path unavailable; "
                  "falling back to fp32 gates", flush=True)
    want = _DTS.get(effective_dt, torch.float32)
    nbytes = 0
    try:
        for i in range(nl):
            if i < first or i % freq != 0:
                continue
            g = f"{prefix}layers.{i}.block_sparse_moe.gate."
            if effective_dt == "int8":
                packed = load_packed(g + "weight")
                if packed is None:
                    raise FileNotFoundError(g + "weight int8 blobs")
                w, scale = packed
                _S[i] = scale
            else:
                w = load(g + "weight").to(want)
            b = load(g + "e_score_correction_bias").to(torch.float32)
            ln = load(f"{prefix}layers.{i}.post_attention_layernorm.weight").to(torch.float32)
            _W[i], _B[i], _LN[i] = w, b, ln
            nbytes += w.numel() * w.element_size() + b.numel() * 4 + ln.numel() * 4
            if i in _S:
                nbytes += _S[i].numel() * _S[i].element_size()
    except Exception as e:                       # never take the run down
        _BROKEN = True
        print(f"[pilot] DISABLED — gate cache build failed: {e!r}", flush=True)
        return
    _CFG.update(top_k=config.num_experts_per_token,
                eps=config.rms_norm_eps, nl=nl, first=first,
                k=_K_ENV or config.num_experts_per_token)
    STATS["gate_bytes"] = nbytes
    STATS["gate_dtype"] = effective_dt
    STATS["build_s"] = time.time() - t0
    _READY = True
    print(f"[pilot] router cache: {len(_W)} gates on {dev} as {effective_dt} "
          f"({nbytes / 2**30:.2f} GB) in {STATS['build_s']:.1f}s | k={_CFG['k']} "
          f"two_step={int(PILOT_TWO)} measure_only={int(MEASURE_ONLY)}", flush=True)


def arm(layer):
    """Capture the input to this layer's post_attention_layernorm — that is the
    pre-MoE hidden state, the thing colibri re-normalises with L+1's post_ln.
    Instance-level patch, so the three shared template objects are each done once."""
    ln = getattr(layer, "post_attention_layernorm", None)
    if ln is None or getattr(ln, "_k3_pilot_armed", False):
        return
    orig = ln.forward

    def fwd(x, _o=orig):
        _CAP["x"] = x
        return _o(x)

    ln.forward = fwd
    ln._k3_pilot_armed = True


def begin_pass(fetch_v2=None):
    """New forward pass: drop predictions from the previous one."""
    if fetch_v2 is not None:
        install_npz(fetch_v2)
    _PRED.clear()
    _PREF.clear()
    _SELF.clear()
    _CAP["x"] = None
    _npz_evict(None)


# --- the router, mirrored from k3pkg/modeling_kimi_linear.py KimiMoEGate.forward
def _route(li, h, k):
    """h: [N, H] fp32 already through layer `li`'s post_attention_layernorm.
    Returns (values [N,k], indices [N,k]).  num_expert_group == 1 for K3, so the
    group-masking branch of KimiMoEGate.forward is inactive and omitted."""
    w = _W[li]
    if w.dtype == torch.int8:
        try:
            logits = torch._weight_int8pack_mm(h.float(), w, _S[li])
        except (NotImplementedError, RuntimeError) as exc:
            # Capability fallback for PyTorch/chip combinations that expose
            # the op but lack a usable MPS kernel. Keep the optimization
            # selectable per chip rather than hard-coding Apple generations.
            global _INT8_WARNED
            if not _INT8_WARNED:
                print(f"[pilot] native int8 gate unavailable ({exc}); "
                      "dequantizing gates lazily", flush=True)
                _INT8_WARNED = True
            w = w.to(torch.float32) * _S.pop(li)[:, None]
            _W[li] = w
            logits = torch.nn.functional.linear(h.float(), w)
    else:
        logits = torch.nn.functional.linear(
            h if h.dtype == w.dtype else h.to(w.dtype), w).float()
    scores = logits.sigmoid()
    scores_for_choice = scores + _B[li].unsqueeze(0)
    return torch.topk(scores_for_choice, k, dim=-1, sorted=False)


def _rms(x, weight, eps):
    """Mirror of KimiRMSNorm.forward (x is fp32 here, so .to(dtype) is a no-op)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf


def on_moe_entry(block, hidden_states, li):
    """Called at the top of layer `li`'s KimiSparseMoeBlock.forward, i.e. after
    layer li's attention and before any expert read.  Predicts layer li+1."""
    global _BROKEN
    if _BROKEN or not _READY:
        return
    nxt = li + 1
    if nxt not in _W and not SELFTEST:
        return
    t0 = time.time()
    try:
        if SELFTEST and li in _W:
            # route layer li with the cached layer-li gate on the model's own
            # post-norm hidden state: must reproduce the model's routing exactly
            _SELF[li] = _route(li, hidden_states.reshape(-1, hidden_states.shape[-1]).float(),
                               _CFG["top_k"])[1].tolist()
        if nxt in _W:
            x = _CAP["x"]
            if x is None:
                return
            x = x.reshape(-1, x.shape[-1])
            if PILOT_TWO:
                sh = getattr(block, "shared_experts", None)
                if sh is not None:
                    y = sh(hidden_states)
                    x = x + y.reshape(-1, y.shape[-1])
            h = _rms(x, _LN[nxt], _CFG["eps"])
            k = _CFG["k"]
            vals, idx = _route(nxt, h, k)
            rows = idx.tolist()
            _PRED[nxt] = rows
            if not MEASURE_ONLY:
                _PREF[nxt] = _union(rows, vals.tolist())
    except Exception as e:
        _BROKEN = True
        print(f"[pilot] DISABLED mid-run at layer {li}: {e!r}", flush=True)
    finally:
        STATS["predict_s"] += time.time() - t0


def _union(rows, vals):
    """Per-position top-k -> one read set.  During prefill the union over T
    positions can be T*k experts; keep the highest-scoring MAX_PREFETCH of them
    so speculation cannot outgrow the pread pool's slot budget."""
    if len(rows) == 1:
        return sorted(set(rows[0]))
    best = {}
    for r, v in zip(rows, vals):
        for e, s in zip(r, v):
            if s > best.get(e, -1e30):
                best[e] = s
    if len(best) <= MAX_PREFETCH:
        return sorted(best)
    return sorted(sorted(best, key=lambda e: -best[e])[:MAX_PREFETCH])


def issue_prefetch(li, fetch_v2, pread=True):
    """Start the predicted reads for layer `li`.  Called by the driver only AFTER
    the current layer's demand reads have completed, so a speculative read can
    never sit in front of a demand read in the pread pool's FIFO queue."""
    ids = _PREF.pop(li, None)
    if not ids:
        return
    STATS["issued_layers"] += 1
    # .bin experts are served by fetch_v2's pread pool, .npz experts by ours;
    # under pread each expert belongs to exactly one of the two, so route each
    # prediction to the path that will actually be asked for it.
    if pread:
        flag = {e: fetch_v2._has_bin(li, e) for e in ids}
        binned = [e for e in ids if flag[e]]
        rest = [e for e in ids if not flag[e]]
    else:
        binned, rest = [], list(ids)
    if BIN and binned:
        STATS["issued_bin"] += len(binned)
        fetch_v2.prefetch_layer(li, binned)
    if NPZ and rest and _orig_cache_load is not None:
        _npz_evict(li)
        n = 0
        with _npz_lock:
            for e in rest:
                k = (li, e)
                if k not in _npz_fut:
                    _npz_fut[k] = _npz_pool.submit(_orig_cache_load, li, e)
                    n += 1
        STATS["issued_npz"] += n


# --- .npz half of the prefetch ----------------------------------------------
_npz_pool = None
_npz_fut = {}                 # (layer, eid) -> Future[_cache_load result]
_npz_lock = threading.Lock()
_orig_cache_load = None


def install_npz(fetch_v2):
    """Memoise fetch_v2._cache_load so a demand load of a predicted .npz expert
    joins the in-flight speculative load instead of starting a second read.
    read_layer() resolves _cache_load out of fetch_v2's module globals at call
    time, so replacing the global is enough."""
    global _npz_pool, _orig_cache_load
    if not NPZ or MEASURE_ONLY or _orig_cache_load is not None:
        return
    _orig_cache_load = fetch_v2._cache_load
    _npz_pool = concurrent.futures.ThreadPoolExecutor(
        NPZ_WORKERS, thread_name_prefix="k3pilotnpz")
    fetch_v2._cache_load = _npz_hook


def _npz_hook(layer, eid):
    """Runs on fetch_v2's worker threads, so the counter is bumped under the lock."""
    with _npz_lock:
        fut = _npz_fut.pop((layer, eid), None)
    if fut is not None:
        try:
            out = fut.result()
        except Exception:
            out = None
        if out is not None:
            with _npz_lock:
                STATS["npz_hits"] += 1
            return out
    return _orig_cache_load(layer, eid)


def _npz_evict(keep_layer):
    """Only one layer is ever predicted ahead, so anything else is a stale
    prediction: drop it. Dropping the future just discards the result; the read
    itself is the wasted I/O a misprediction is allowed to cost."""
    if _orig_cache_load is None:
        return
    with _npz_lock:
        stale = [k for k in _npz_fut if k[0] != keep_layer]
        for k in stale:
            _npz_fut.pop(k).cancel()
    STATS["npz_wasted"] += len(stale)


def on_actual(li, rows):
    """`rows`: the model's real per-position top-k ids at layer li.  Scores the
    prediction that was made one layer earlier."""
    pred = _PRED.pop(li, None)
    if pred is not None and len(pred) == len(rows):
        rec = _REC.setdefault(li, [0, 0])
        dec = _REC_DEC.setdefault(li, [0, 0]) if len(rows) == 1 else None
        for a, p in zip(rows, pred):
            m, n = len(set(a) & set(p)), len(a)
            rec[0] += m
            rec[1] += n
            if dec is not None:
                dec[0] += m
                dec[1] += n
    if SELFTEST:
        sp = _SELF.pop(li, None)
        if sp is not None and len(sp) == len(rows):
            rec = _REC_SELF.setdefault(li, [0, 0])
            for a, p in zip(rows, sp):
                rec[0] += len(set(a) & set(p))
                rec[1] += len(a)


# --- fetch_v2 interop --------------------------------------------------------
def install_async_drain(reader):
    """fetch_v2._PreadReader._drain() waits out mispredicted in-flight reads
    before the demand misses of the same layer are submitted.  With K3_PILOT
    there are ~k*(1-recall) of those per layer; move the wait off the critical
    path.  Slots are still only returned to the free list after their read
    actually finishes, so no buffer is ever recycled under a live read."""
    if getattr(reader, "_k3_pilot_async", False):
        return
    orig = reader._drain

    def drain(jobs, _o=orig):
        threading.Thread(target=_o, args=(jobs,), daemon=True).start()

    reader._drain = drain
    reader._k3_pilot_async = True


# --- reporting ---------------------------------------------------------------
def _pct(rec):
    h = sum(v[0] for v in rec.values())
    t = sum(v[1] for v in rec.values())
    return (100.0 * h / t if t else 0.0), h, t


def report(fetch_stats=None):
    if not _REC and not _REC_SELF:
        return ""
    overall, h, t = _pct(_REC)
    per = [(li, 100.0 * v[0] / v[1]) for li, v in sorted(_REC.items()) if v[1]]
    lo = min(per, key=lambda r: r[1]) if per else (-1, 0.0)
    hi = max(per, key=lambda r: r[1]) if per else (-1, 0.0)
    out = [f"[pilot] top-{_CFG.get('k')} recall one layer ahead: {overall:.1f}% "
           f"({h}/{t} over {len(per)} layers, all passes) | two_step={int(PILOT_TWO)} "
           f"| worst L{lo[0]} {lo[1]:.1f}% best L{hi[0]} {hi[1]:.1f}%"]
    if _REC_DEC:
        d, dh, dt = _pct(_REC_DEC)
        out.append(f"[pilot] top-{_CFG.get('k')} recall, DECODE passes only "
                   f"(T=1): {d:.1f}% ({dh}/{dt})")
    out.append("[pilot] per-layer recall %: " +
               " ".join(f"{li}:{p:.0f}" for li, p in per))
    if _REC_SELF:
        s, sh, st = _pct(_REC_SELF)
        out.append(f"[pilot] selftest (cached gate vs model's own routing, same "
                   f"layer): {s:.2f}% ({sh}/{st}) — must be 100.00%")
    out.append(f"[pilot] speculative reads issued over {STATS['issued_layers']} layers: "
               f"{STATS['issued_bin']} .bin + {STATS['issued_npz']} .npz | predict "
               f"{STATS['predict_s']:.1f}s | gate cache {STATS['gate_bytes'] / 2**30:.2f} GB")
    out.append(f"[pilot] npz path: hits={STATS['npz_hits']} wasted={STATS['npz_wasted']}")
    if fetch_stats:
        out.append(f"[pilot] pread pool: prefetch_hits={fetch_stats.get('prefetch_hits', 0)} "
                   f"prefetch_wasted={fetch_stats.get('prefetch_wasted', 0)} "
                   f"slots={fetch_stats.get('pread_slots', 0)} "
                   f"experts_read={fetch_stats.get('pread_experts', 0)}")
    return "\n".join(out)
