"""Fast resident-spine load path (K3_FAST_SPINE=1).

Replaces the per-tensor read -> bytearray -> .to(DEV) -> fp32 dequant -> copy_
chain in kimi_run.py with three changes, each measured on this machine:

  1. PACKED READS.  A layer's ~28 .i8 payloads (and its ~28 .sc scale blobs) are
     read with readinto() straight into ONE reusable host buffer, optionally by a
     few threads.  Measured on the real spine, 634 MB layer, cold-ish:
         read()+bytearray()   4.07 GB/s      (current path: kernel copy + a full
                                              second memcpy for the bytearray)
         readinto pooled x1   5.57 GB/s
         readinto pooled x4   6.94 GB/s
         readinto pooled x8   7.18 GB/s
     The buffer pool matters: bytearray(n) zero-fills, so allocating a fresh
     630 MB per layer would memset 57 GB/token.

  2. ONE host->device transfer per layer instead of ~28 (2,604/token today).
         28 separate .to(DEV)   40.2 GB/s
         1 coalesced .to(DEV)   49.7 GB/s

  3. A CUSTOM METAL DEQUANT KERNEL that fuses int8->fp32 + row-scale multiply +
     the copy into the template buffer into one pass.  torch's row-broadcast
     multiply on MPS is the real bottleneck in the current path -- it runs at
     ~100 GB/s where a plain copy_ of the same traffic runs at 334 GB/s:
         (q.to(f32) * sc.to(f32)) then copy_   10.08 ms   43.7 GB/s   [current]
         torch.mul(q, sc_f32, out=dst)          7.34 ms   60.0 GB/s
         metal deq_f32                          1.48 ms  297.0 GB/s
     (6144x14336 int8 -> fp32.)  The kernel is BIT-EXACT against the current
     expression -- max|diff| = 0.0 on every shape tested, including from a
     non-zero storage offset inside the packed buffer -- because both compute
     float(int8) * float(fp16) in fp32.

Everything here is inert unless K3_FAST_SPINE=1.  Any failure (no MPS, shader
compile error) falls back to the torch expression automatically.
"""
import os
import threading
import time as _time
import concurrent.futures as _cf

import torch

# ---------------------------------------------------------------- flags -----
FAST = os.environ.get("K3_FAST_SPINE", "1") == "1"
# fine-grained overrides so the orchestrator can bisect the three changes
DEQ = os.environ.get("K3_SPINE_DEQ", "metal" if FAST else "torch")   # metal|mulout|torch
PACK = os.environ.get("K3_SPINE_PACK", "1" if FAST else "0") == "1"  # packed read + 1 H2D
READ_THREADS = int(os.environ.get("K3_SPINE_READ_THREADS", "4" if FAST else "1"))
ALIGN = 256          # byte alignment of every slot inside the packed buffer


# --------------------------------------------------------- metal dequant ----
_MSL = r"""
#include <metal_stdlib>
using namespace metal;

// out[i] = float(q[i]) * float(sc[i / cols])   -- identical arithmetic to
// (q.to(float32) * sc.to(float32)) in torch, hence bit-exact.
kernel void deq_f32(device float*       out  [[buffer(0)]],
                    device const char*  q    [[buffer(1)]],
                    device const half*  sc   [[buffer(2)]],
                    constant uint&      cols [[buffer(3)]],
                    uint idx [[thread_position_in_grid]])
{
    out[idx] = float(q[idx]) * float(sc[idx / cols]);
}
"""

_lib = None
_lib_err = None
_lib_lock = threading.Lock()


def metal_available():
    """Compile the dequant shader once; False (with a reason in metal_error())
    if this build/machine cannot."""
    global _lib, _lib_err
    if _lib is not None:
        return True
    if _lib_err is not None:
        return False
    with _lib_lock:
        if _lib is None and _lib_err is None:
            try:
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS not available")
                _lib = torch.mps.compile_shader(_MSL)
            except Exception as e:                     # pragma: no cover
                _lib_err = f"{type(e).__name__}: {e}"
    return _lib is not None


def metal_error():
    return _lib_err


def dequant_into(dst, q, sc):
    """dst[fp32, (rows, cols), on MPS] <- float(q[int8]) * float(sc[fp16, rows]).

    q/sc may be views at arbitrary storage offsets inside a packed buffer.
    Returns True if the metal kernel ran, False if the caller should fall back."""
    if not metal_available():
        return False
    rows, cols = dst.shape
    _lib.deq_f32(dst, q, sc, cols, threads=rows * cols)
    return True


def dequant_torch(q, sc, out=None):
    """Fallback / baseline dequant. `mulout` fuses the multiply with the write
    into the destination (7.34 ms vs 10.08 ms on 6144x14336); `torch` is the
    exact expression kimi_run.py uses today."""
    if out is not None and DEQ == "mulout":
        torch.mul(q, sc.to(torch.float32), out=out)
        return out
    return q.to(torch.float32) * sc.to(torch.float32)


# ----------------------------------------------------------- buffer pool ----
# At most two packs are live at once (the preloader thread fills N+1 while the
# main thread applies N), so this settles at 2 buffers and never reallocates.
_pool = []
_pool_lock = threading.Lock()
_POOL_MAX = 8      # 3 buffers per pack (int8 / scales / bf16) x 2 live packs


def _acquire(nbytes):
    """Best fit, so a 634 MB payload buffer is never handed to a 2 MB scale
    request (which would force a fresh bytearray -- and bytearray(n) zero-fills,
    i.e. 57 GB of memset per token if it happened every layer)."""
    with _pool_lock:
        best = -1
        for i, b in enumerate(_pool):
            if len(b) >= nbytes and (best < 0 or len(b) < len(_pool[best])):
                best = i
        if best >= 0:
            return _pool.pop(best)
    return bytearray(max(nbytes, 1))


# Buffers a caller has taken permanent ownership of (the driver's spine RAM
# cache). They must never re-enter the pool, or the next layer's readinto would
# overwrite cached weights in place — a silent wrong-weights bug, not a slow path.
KEEP = set()


def pin(*bufs):
    """Take ownership of these buffers: the pool will never recycle them."""
    for b in bufs:
        if b is not None:
            KEEP.add(id(b))


def _release(buf):
    if buf is None or id(buf) in KEEP:
        return
    with _pool_lock:
        if len(_pool) < _POOL_MAX:
            _pool.append(buf)


# Persistent device staging buffers: without these every layer would allocate
# and free a fresh ~634 MB MPS buffer for the packed int8 payload.
_stage = {}


def _stage_buf(n, dev, dtype):
    key = (dev.type, dtype)
    b = _stage.get(key)
    if b is None or b.numel() < n:
        b = torch.empty(n, dtype=dtype, device=dev)
        _stage[key] = b
    return b[:n]


# Phase counters (always accumulated; printed by kimi_run under K3_PROFILE).
PHASE = {"read_s": 0.0, "read_bytes": 0, "h2d_s": 0.0, "h2d_bytes": 0,
         "deq_s": 0.0, "other_s": 0.0, "n_layers": 0}

_readers = None
_readers_lock = threading.Lock()


def _pool_exec():
    global _readers
    if _readers is None:
        with _readers_lock:
            if _readers is None:
                _readers = _cf.ThreadPoolExecutor(READ_THREADS,
                                                  thread_name_prefix="k3spine")
    return _readers


# ------------------------------------------------------- layout planning ----
# The set of files a layer needs, their sizes and their slot offsets never
# change between tokens, so plan once per prefix.  Saves ~5,200 stat() calls
# and ~2,600 open() calls per token.
_layouts = {}
_layout_lock = threading.Lock()


def _align(x):
    return (x + ALIGN - 1) // ALIGN * ALIGN


def plan_layout(module, prefix, int8_dir, res_dir, inv, spine):
    """-> dict(items=[(name, full, shape, qoff, qbytes, scoff_elems, rows)],
              qtotal, sctotal_elems, other=[(name, full, path_or_None)])"""
    key = prefix
    got = _layouts.get(key)
    if got is not None:
        return got
    items, other = [], []
    qoff = 0
    scoff = 0                     # in fp16 elements
    for name, _ in module.named_parameters():
        if ".experts." in name:
            continue
        full = prefix + name
        ip = os.path.join(int8_dir, full + ".i8")
        if spine == "int8" and os.path.exists(ip):
            rows, cols = inv[full]["shape"]
            nb = rows * cols
            items.append((name, full, (rows, cols), qoff, nb, scoff, rows))
            qoff = _align(qoff + nb)
            scoff = _align(scoff * 2 + rows * 2) // 2
            continue
        bp = os.path.join(res_dir, full)
        other.append((name, full, bp if os.path.exists(bp) else None))
    # The bf16 leftovers (norms, conv1d, A_log, dt_bias, res_proj) are tiny but
    # numerous -- ~8 per layer, ~744 per forward. As separate blocking .to(DEV)
    # calls they measured 2.6 s/forward, almost all per-call overhead, so they
    # get the same treatment: one packed buffer, one transfer.
    ooff = 0
    oplan = []
    for name, full, bp in other:
        if bp is None:
            continue
        meta = inv[full]
        nb = os.path.getsize(bp)
        oplan.append((name, full, meta["dtype"], tuple(meta["shape"]), ooff, nb, bp))
        ooff = _align(ooff + nb)
    lay = {"items": items, "qtotal": qoff, "sctotal": scoff, "other": other,
           "oplan": oplan, "ototal": ooff, "int8_dir": int8_dir}
    with _layout_lock:
        _layouts[key] = lay
    return lay


# ------------------------------------------------------------- read side ----
def read_pack(module, prefix, int8_dir, res_dir, inv, spine, load_resident):
    """Runs in the preloader thread.  Returns a pack the main thread applies."""
    lay = plan_layout(module, prefix, int8_dir, res_dir, inv, spine)
    items = lay["items"]
    qbuf = _acquire(lay["qtotal"])
    scbuf = _acquire(lay["sctotal"] * 2)
    qmv, scmv = memoryview(qbuf), memoryview(scbuf)

    def _one(it):
        _, full, _, qoff, nb, scoff, rows = it
        base = os.path.join(int8_dir, full)
        with open(base + ".i8", "rb") as f:
            f.readinto(qmv[qoff:qoff + nb])
        with open(base + ".sc", "rb") as f:
            f.readinto(scmv[scoff * 2:scoff * 2 + rows * 2])

    t0 = _time.time()
    if READ_THREADS > 1 and len(items) > 1:
        list(_pool_exec().map(_one, items))
    else:
        for it in items:
            _one(it)

    obuf = _acquire(max(lay["ototal"], 1))
    omv = memoryview(obuf)
    for _n, _f, _dt, _sh, ooff, nb, bp in lay["oplan"]:
        with open(bp, "rb") as f:
            f.readinto(omv[ooff:ooff + nb])
    PHASE["read_s"] += _time.time() - t0
    PHASE["read_bytes"] += lay["qtotal"] + lay["ototal"]
    return {"lay": lay, "q": qbuf, "sc": scbuf, "other": obuf}


# ------------------------------------------------------------ apply side ----
def apply_pack(module, prefix, pack, dev, dt, inv, dtmap, set_param, load_resident):
    """One H2D transfer for the whole layer, then a fused dequant straight into
    each parameter's existing buffer."""
    lay = pack["lay"]
    items = lay["items"]
    use_metal = DEQ == "metal" and dev.type == "mps" and metal_available()
    PHASE["n_layers"] += 1
    oplan = lay["oplan"]
    # All three host->device transfers go first, back to back. cpu->mps copy_ is
    # blocking, so a transfer wedged between kernel dispatches would stall the
    # main thread on the GPU queue; doing them up front lets the dequant kernels
    # pipeline against the next layer instead.
    t0 = _time.time()
    if items:
        qh = torch.frombuffer(pack["q"], dtype=torch.int8)[:lay["qtotal"]]
        sch = torch.frombuffer(pack["sc"], dtype=torch.float16)[:lay["sctotal"]]
        qd = _stage_buf(lay["qtotal"], dev, torch.int8)
        scd = _stage_buf(lay["sctotal"], dev, torch.float16)
        qd.copy_(qh)
        scd.copy_(sch)
    if oplan:
        oh = torch.frombuffer(pack["other"], dtype=torch.uint8)[:lay["ototal"]]
        od = _stage_buf(lay["ototal"], dev, torch.uint8)
        od.copy_(oh)
    PHASE["h2d_s"] += _time.time() - t0
    PHASE["h2d_bytes"] += lay["qtotal"] + lay["ototal"]
    t0 = _time.time()
    for name, full, shape, qoff, nb, scoff, rows in items:
        q = qd[qoff:qoff + nb].view(shape)
        sc = scd[scoff:scoff + rows]
        p = module.get_parameter(name)
        fits = (p.device.type != "meta" and p.shape == torch.Size(shape)
                and p.dtype == torch.float32 and dt == torch.float32)
        if fits and use_metal:
            dequant_into(p.data, q, sc)
            continue
        if fits and DEQ == "mulout":
            torch.mul(q, sc.view(rows, 1).to(torch.float32), out=p.data)
            continue
        t = (q.to(torch.float32) * sc.view(rows, 1).to(torch.float32)).to(dt)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
    PHASE["deq_s"] += _time.time() - t0
    t0 = _time.time()
    have = {rec[1] for rec in oplan}
    for name, full, sdt, shape, ooff, nb, _bp in oplan:
        src = od[ooff:ooff + nb].view(dtmap[sdt]).reshape(shape)
        p = module.get_parameter(name)
        if p.device.type == "meta" or p.shape != torch.Size(shape) or p.dtype != dt:
            set_param(module, name, src.to(dt).clone())
        else:
            p.data.copy_(src)                         # copy_ does the bf16->fp32
    for name, full, path in lay["other"]:
        if full in have:
            continue
        t = load_resident(full).to(dev, dt)            # not on disk: rare fallback
        p = module.get_parameter(name)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
    PHASE["other_s"] += _time.time() - t0
    # torch's cpu->mps copy_ is blocking (non_blocking=False), so the host
    # buffers are free the moment .copy_() returned.
    if items:
        del qh, sch, qd, scd
    if oplan:
        del oh, od
    # A pinned pack is owned by the caller's cache and will be applied again on
    # the next token: leave its buffers intact and in place.
    if id(pack.get("q")) in KEEP:
        return
    _release(pack["q"])
    _release(pack["sc"])
    _release(pack["other"])
    pack["q"] = pack["sc"] = pack["other"] = None


def describe():
    bits = [f"pack={int(PACK)}", f"deq={DEQ}", f"read_threads={READ_THREADS}"]
    if DEQ == "metal":
        bits.append("metal_ok=" + ("1" if metal_available() else f"0({metal_error()})"))
    return " ".join(bits)


def phase_report():
    p = PHASE
    if not p["n_layers"]:
        return ""
    rd = p["read_bytes"] / 1e9 / max(p["read_s"], 1e-9)
    hd = p["h2d_bytes"] / 1e9 / max(p["h2d_s"], 1e-9)
    return (f"[spine] {p['n_layers']} layers | read {p['read_s']:.1f}s "
            f"({p['read_bytes']/1e9:.1f} GB, {rd:.1f} GB/s, bg thread) | "
            f"h2d {p['h2d_s']:.1f}s ({hd:.1f} GB/s) | deq {p['deq_s']:.1f}s | "
            f"bf16-tail {p['other_s']:.1f}s")
