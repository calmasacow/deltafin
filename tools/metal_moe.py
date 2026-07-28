"""Metal MoE expert path: the whole layer's selected experts (MXFP4 dequant + w1/w3
GEMV + SiTU + w2 GEMV + weighted sum) run on the GPU from one command buffer.

Drop-in for tools/fast_moe.py's moe_infer_fast — same signature, same semantics —
so it can be swapped in behind a flag:

    import metal_moe
    moe = metal_moe.moe_infer if metal_moe.available() else fast_moe.moe_infer_fast

which is exactly what tools/kimi_run.py does behind K3_MOE=cpu|metal.  moe_infer()
is moe_infer_metal() plus the optional K3_MOE_CHECK cross-check against the CPU
kernel; use moe_infer_metal() directly to bypass that entirely.

Build the backing dylib first (nothing in any Makefile is touched):

    clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib \
        tools/metal_moe.mm -framework Metal -framework Foundation \
        -o tools/libk3metalmoe.dylib

Zero-copy contract
------------------
An expert reaches the GPU as its verbatim 17,547,264-byte shard span.  When the six
arrays in `raw_experts[e]` are already contiguous inside one buffer at the on-disk
offsets — which is exactly what fetch_v2's mmap'd L*-E*.bin cache gives you — the
base pointer is handed to Metal and wrapped with newBufferWithBytesNoCopy: no copy,
no staging, the GPU reads the page cache directly.  EXPERT_SPAN is 1071 * 16 KB, so
an mmap base is always correctly aligned.

Those wraps are cached in the dylib keyed by pointer, so this module pins the owning
numpy/mmap object for as long as the wrap lives (LRU, K3_METAL_PIN_MAX experts,
default 512 -> ~9 GB of *virtual* mapping, resident only where touched).  Eviction
calls k3_metal_drop() before releasing the pin, so a wrap never outlives its memory.

Experts that arrive as separate allocations (the legacy .npz cache path) are packed
into a page-aligned anonymous mmap slot first — one memcpy, then still zero-copy.

Buffer recycling (tools/fetch_v2.py) — the correctness invariant
---------------------------------------------------------------
fetch_v2's pread reader owns a pool of recyclable page-aligned EXPERT_SPAN slots:
the same base pointer carries a DIFFERENT expert a couple of layers later, and a
surplus slot can be unmapped outright (_PreadReader._give trims to
K3_PREAD_MAX_FREE).  A pointer-keyed wrap cache is only safe under one invariant,
which this module now enforces explicitly:

    a cached wrap exists  =>  this module holds a strong reference to the object
                              that owns its mapping.

Consequences, both of which are load-bearing:
  * the mapping cannot be unmapped while the wrap lives, so the pointer cannot be
    handed to a *different* mapping behind Metal's back;
  * therefore `_pins[ptr] is owner` is a sound identity test — if the owner object
    ever differs, the memory genuinely changed hands and `_sync_wrap` drops the
    stale wrap (k3_metal_drop) before the dylib can reuse it.

Same pointer + same owner + new contents (a recycled slot re-pread with another
expert) needs no invalidation at all: the wrap aliases the live pages, so the GPU
reads whatever the host last wrote there.  MTLResourceStorageModeShared on Apple
Silicon is coherent, and every call is commit + waitUntilCompleted, so the pread
that filled the slot always happens-before the dispatch that reads it.

Pin eviction is the release valve: dropping a pin does NOT unmap a fetch_v2 slot
(fetch_v2 still owns it), it just lets fetch_v2 unmap it later — and by then the
wrap is already gone.  Proof harness: tools/test_metal_recycle.py.
"""
import ctypes
import mmap
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DYLIB = os.environ.get("K3_METAL_LIB", os.path.join(_HERE, "libk3metalmoe.dylib"))
_MSL = os.environ.get("K3_METAL_SRC", os.path.join(_HERE, "metal", "moe_mxfp4.metal"))

HIDDEN, INTER = 3584, 3072
_P, _S = 5505024, 344064
EXPERT_SPAN = 3 * (_P + _S)                                  # 17,547,264
# (tensor, tuple slot, byte offset, shape) inside the span — tools/fetch_v2.py LAYOUT
_OFFSETS = (("w1", 0, 0, (3072, 1792)), ("w1", 1, _P, (3072, 112)),
            ("w2", 0, _P + _S, (3584, 1536)), ("w2", 1, 2 * _P + _S, (3584, 96)),
            ("w3", 0, 2 * (_P + _S), (3072, 1792)), ("w3", 1, 3 * _P + 2 * _S, (3072, 112)))
_SIZES = {0: _P, 1: _S}

PIN_MAX = int(os.environ.get("K3_METAL_PIN_MAX", "512"))
# cross-check the first N MoE calls against the CPU kernel (tools/fast_moe.py) and
# raise if they disagree. 0 = off (default) and costs nothing.
CHECK = int(os.environ.get("K3_MOE_CHECK", "0"))
CHECK_TOL = float(os.environ.get("K3_MOE_CHECK_TOL", "1e-4"))
# Escape hatch for tools/test_metal_recycle.py --unsafe ONLY: pretend a pointer is
# a permanent expert identity (no owner pin, no invalidation), i.e. the failure
# mode the pin protocol exists to prevent. Never set this in a real run.
STALE_OK = os.environ.get("K3_METAL_STALE_OK", "0") == "1"

_lib = None
_load_error = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(_DYLIB)
    except OSError as e:
        _load_error = f"{_DYLIB}: {e}"
        return None
    lib.k3_metal_init.argtypes = [ctypes.c_char_p]
    lib.k3_metal_init.restype = ctypes.c_int
    lib.k3_metal_available.restype = ctypes.c_int
    lib.k3_metal_last_error.restype = ctypes.c_char_p
    lib.k3_metal_moe_layer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float)]
    lib.k3_metal_moe_layer.restype = ctypes.c_int
    lib.k3_metal_drop.argtypes = [ctypes.c_void_p]
    lib.k3_metal_flush.argtypes = []
    lib.k3_metal_shapes.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_longlong)]
    lib.k3_metal_stats.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    lib.k3_metal_set_mode.argtypes = [ctypes.c_int]
    lib.k3_metal_set_mode.restype = ctypes.c_int
    lib.k3_metal_mode.restype = ctypes.c_int
    if lib.k3_metal_init(_MSL.encode()) != 0:
        _load_error = lib.k3_metal_last_error().decode()
        return None
    h, i, span = ctypes.c_int(), ctypes.c_int(), ctypes.c_longlong()
    lib.k3_metal_shapes(ctypes.byref(h), ctypes.byref(i), ctypes.byref(span))
    if (h.value, i.value, span.value) != (HIDDEN, INTER, EXPERT_SPAN):
        _load_error = (f"shape mismatch: dylib {h.value}/{i.value}/{span.value} "
                       f"vs python {HIDDEN}/{INTER}/{EXPERT_SPAN}")
        return None
    _lib = lib
    return _lib


def available():
    """True if a Metal device came up and the MSL compiled. Never raises."""
    lib = _load()
    return bool(lib and lib.k3_metal_available())


def last_error():
    if _load_error:
        return _load_error
    lib = _load()
    return lib.k3_metal_last_error().decode() if lib else "not loaded"


def stats():
    """{calls, zero_copy_wraps, copies, cache_entries, bindless} from the dylib."""
    lib = _load()
    if not lib:
        return {}
    buf = (ctypes.c_longlong * 5)()
    lib.k3_metal_stats(buf)
    keys = ("calls", "zero_copy_wraps", "copies", "cache_entries", "bindless")
    d = dict(zip(keys, list(buf)))
    d["pinned"] = len(_pins)
    d["rewraps"] = _rewraps            # stale wraps dropped on owner change
    return d


def set_mode(bindless):
    """1 = one dispatch per stage for the whole layer (Tier-2 argument buffer),
    0 = one dispatch per expert inside one concurrent encoder. Returns mode in effect."""
    lib = _load()
    return lib.k3_metal_set_mode(1 if bindless else 0) if lib else -1


# --- span assembly ---------------------------------------------------------
_pins = {}          # ptr -> owning mapping object (strong ref), insertion-ordered = LRU
_perm = set()       # ptrs of module-owned staging slots: never evicted
_slots = []         # page-aligned anonymous mmaps for non-contiguous experts
_slot_ptr = []
_rewraps = 0        # times a pointer changed owner and the stale wrap was dropped


def _owner(a):
    """The object that actually owns `a`'s memory: follow the numpy .base chain to
    its root (an mmap.mmap, a bytes, another ndarray...).  Holding this alive is
    what keeps the pages mapped — and it is a stable identity for one mapping, so
    two different mappings can never compare equal."""
    o = a
    for _ in range(8):
        b = getattr(o, "base", None)
        if b is None:
            return o
        o = b
    return o


def _unpin(ptr):
    """Release a wrap, then the memory behind it — never the other way round: a
    cached wrap must never outlive its mapping, or a later mmap that lands on the
    same address would silently be read as the old expert."""
    lib = _load()
    if lib:
        lib.k3_metal_drop(ctypes.c_void_p(ptr))
    _pins.pop(ptr, None)


def _sync_wrap(ptr, owner):
    """Make the dylib's pointer-keyed wrap cache agree with reality for `ptr`, and
    take the strong reference that keeps it true.

    Called BEFORE the dispatch, so the reference is held for the whole GPU call.
    If `ptr` was previously wrapped over a different mapping (only reachable once
    the earlier pin was evicted and fetch_v2 then unmapped and re-mapped a slot at
    the same address) the stale wrap is dropped first, so the dylib re-wraps the
    memory that is live now instead of resurrecting dead pages."""
    global _rewraps
    if STALE_OK:                             # test-only: reproduce the naive cache
        return
    prev = _pins.pop(ptr, None)              # pop = refresh LRU recency
    if prev is not None and prev is not owner:
        lib = _load()
        if lib:
            lib.k3_metal_drop(ctypes.c_void_p(ptr))
        _rewraps += 1
    _pins[ptr] = owner


def _trim(protect):
    """Evict pins down to the cap. `protect` (this call's pointers) and the
    permanent staging slots are never evicted, so the cap floors at len(protect)."""
    cap = max(PIN_MAX, len(protect))
    while len(_pins) > cap:
        victim = next((p for p in _pins if p not in protect and p not in _perm), None)
        if victim is None:
            break
        _unpin(victim)


def _slot(i):
    """A page-aligned EXPERT_SPAN staging buffer (anonymous mmap -> always aligned)."""
    while len(_slots) <= i:
        m = mmap.mmap(-1, EXPERT_SPAN)
        arr = np.ndarray((EXPERT_SPAN,), dtype=np.uint8, buffer=m)   # writable view
        _slots.append((m, arr))
        _slot_ptr.append(arr.ctypes.data)
        _perm.add(arr.ctypes.data)
    return _slots[i][1], _slot_ptr[i], _slots[i][0]


def _span_ptr(raw, slot):
    """(pointer to the expert's contiguous 17,547,264-byte span, its owning
    mapping object).  The owner is never None: staged spans are owned by the
    module's permanent mmap slots, zero-copy spans by whatever fetch_v2 handed us
    (a recycled pread slot's mmap, an np.memmap, or the HTTP path's bytes)."""
    base = raw["w1"][0].ctypes.data
    ok = True
    for w, k, off, _shape in _OFFSETS:
        a = raw[w][k]
        if (a.dtype != np.uint8 or not a.flags["C_CONTIGUOUS"]
                or a.nbytes != _SIZES[k] or a.ctypes.data != base + off):
            ok = False
            break
    if ok:
        # alignment is the dylib's call: page-aligned -> cached zero-copy wrap,
        # otherwise its own scratch buffer (one memcpy in C, never pointer-cached).
        return base, _owner(raw["w1"][0])
    dst, ptr, own = _slot(slot)
    for w, k, off, _shape in _OFFSETS:
        a = raw[w][k]
        if a.dtype != np.uint8 or a.nbytes != _SIZES[k]:
            raise ValueError(f"expert tensor {w}[{k}] is {a.dtype}/{a.nbytes} B, "
                             f"expected uint8/{_SIZES[k]} B")
        dst[off:off + _SIZES[k]] = np.ascontiguousarray(a).reshape(-1)
    return ptr, own


def flush():
    """Drop every cached GPU wrap and unpin the memory behind it."""
    lib = _load()
    if lib:
        lib.k3_metal_flush()
    _pins.clear()
    _spans.clear()


# --- persistent expert mmaps (optional, but this is what makes the wrap cache pay)
# The dylib caches zero-copy wraps by pointer.  fetch_v2 creates a FRESH np.memmap
# on every fetch_experts() call, so each token would land on a new address and
# re-wrap.  Going through raw_from_cache() keeps one mmap per (layer, eid) alive, so
# the pointer — and therefore the MTLBuffer — is stable for the expert's lifetime.
ECACHE = os.path.join(os.path.dirname(_HERE), "k3-experts")
_spans = {}          # (layer, eid) -> (memmap, raw dict); insertion-ordered = LRU


def raw_from_cache(layer, eid, root=None):
    """{w1|w2|w3: (packed, scale)} backed by a persistent mmap of the .bin cache
    file, or None if that expert is not on disk as a raw span.  Same dict shape
    fetch_v2.fetch_experts(dequant=False) returns, so it drops straight in."""
    key = (layer, eid)
    hit = _spans.get(key)
    if hit is not None:
        _spans[key] = _spans.pop(key)
        return hit[1]
    path = os.path.join(root or ECACHE, f"L{layer}-E{eid}.bin")
    if not os.path.exists(path) or os.path.getsize(path) != EXPERT_SPAN:
        return None
    m = np.memmap(path, dtype=np.uint8, mode="r")
    raw = {}
    for w, k, off, shape in _OFFSETS:
        raw.setdefault(w, [None, None])[k] = m[off:off + _SIZES[k]].reshape(shape)
    raw = {w: (v[0], v[1]) for w, v in raw.items()}
    _spans[key] = (m, raw)
    while len(_spans) > PIN_MAX and len(_spans) > 1:
        old = next(iter(_spans))
        victim = _spans.pop(old)                 # held until after the wrap is gone
        _unpin(victim[1]["w1"][0].ctypes.data)
        del victim
    return raw


# --- public API ------------------------------------------------------------
def moe_infer_metal(x, topk_ids, topk_weight, raw_experts):
    """x: torch [N, 3584] fp32; returns torch [N, 3584] fp32.
    Same contract as fast_moe.moe_infer_fast: sum_k w_k * expert_k(x), with
    raw_experts mapping expert id -> {w1|w2|w3: (packed u8, scale u8)}.
    One command buffer per token, covering that token's whole selected set."""
    lib = _load()
    if lib is None:
        raise RuntimeError(f"metal MoE unavailable: {last_error()}")
    xnp = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
    N, H = xnp.shape
    if H != HIDDEN:
        raise ValueError(f"x has hidden {H}, kernel is built for {HIDDEN}")
    out = np.zeros((N, H), dtype=np.float32)
    ids = topk_ids.tolist()
    ws = topk_weight.to(torch.float32).tolist()
    fp = ctypes.POINTER(ctypes.c_float)
    for t in range(N):
        eids, wts = ids[t], ws[t]
        n = len(eids)
        ptrs = (ctypes.c_void_p * n)()
        protect = set()
        for i, e in enumerate(eids):
            p, obj = _span_ptr(raw_experts[e], i)
            ptrs[i] = p
            # BEFORE the dispatch: drop any wrap left over from a different
            # mapping at this address, and hold the owner for the whole call.
            _sync_wrap(p, obj)
            protect.add(p)
        wbuf = np.ascontiguousarray(wts, dtype=np.float32)
        xt = np.ascontiguousarray(xnp[t])
        ot = out[t]                                  # view; out owns the memory
        rc = lib.k3_metal_moe_layer(ptrs, n,
                                    wbuf.ctypes.data_as(fp),
                                    xt.ctypes.data_as(fp),
                                    ot.ctypes.data_as(fp))
        if rc != 0:
            raise RuntimeError(f"k3_metal_moe_layer rc={rc}: {last_error()}")
        _trim(protect)                               # never evicts this call's set
    return torch.from_numpy(out).to(x.device, x.dtype)


# --- optional CPU cross-check (K3_MOE_CHECK=N) ------------------------------
_checked = 0
_check_worst = 0.0


def check_stats():
    return {"checked": _checked, "worst_rel": _check_worst,
            "rewraps": _rewraps, "pins": len(_pins)}


def moe_infer(x, topk_ids, topk_weight, raw_experts):
    """moe_infer_metal, plus a CPU cross-check on the first K3_MOE_CHECK calls.

    The check runs on the SAME raw_experts buffers the GPU just read, in the same
    layer, so it is exactly the assertion a recycled-slot aliasing bug would have
    to survive: a stale wrap computes a different expert and the relative error
    blows past CHECK_TOL immediately."""
    global _checked, _check_worst
    out = moe_infer_metal(x, topk_ids, topk_weight, raw_experts)
    if _checked < CHECK:
        _checked += 1
        import fast_moe
        ref = fast_moe.moe_infer_fast(x, topk_ids, topk_weight, raw_experts)
        a = out.detach().to("cpu", torch.float32).numpy()
        b = ref.detach().to("cpu", torch.float32).numpy()
        den = max(float(np.abs(b).max()), 1e-30)
        rel = float(np.abs(a - b).max()) / den
        _check_worst = max(_check_worst, rel)
        if rel > CHECK_TOL:
            raise RuntimeError(
                f"K3_MOE_CHECK: metal vs cpu rel {rel:.3e} > {CHECK_TOL:g} "
                f"on call {_checked} (ids={topk_ids.view(-1).tolist()})")
    return out
