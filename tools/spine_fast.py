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
import types

import torch

import spine_io

# ---------------------------------------------------------------- flags -----
FAST = os.environ.get("K3_FAST_SPINE", "1") == "1"
# fine-grained overrides so the orchestrator can bisect the three changes
DEQ = os.environ.get("K3_SPINE_DEQ", "metal" if FAST else "torch")   # metal|mulout|torch
PACK = os.environ.get("K3_SPINE_PACK", "1" if FAST else "0") == "1"  # packed read + 1 H2D
READ_THREADS = int(os.environ.get("K3_SPINE_READ_THREADS", "4" if FAST else "1"))
ALIGN = 256          # byte alignment of every slot inside the packed buffer

spine_io.set_process_io_policy()


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


def unpin(*bufs):
    """Hand ownership back.  ONLY call this after the pack has been dropped from
    whatever cache pinned it -- an unpinned buffer can be recycled by the next
    layer's read, which would silently rewrite it in place."""
    for b in bufs:
        if b is not None:
            KEEP.discard(id(b))


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


# ------------------------------------------------ dynamic packed-q8 Q/K/V ---
_QKV_NAMES = {
    "self_attn.q_proj.weight": "q",
    "self_attn.k_proj.weight": "k",
    "self_attn.v_proj.weight": "v",
}
_QKV_ROLES = tuple(_QKV_NAMES.values())


class DynamicQ8State:
    """Shared kill switch for the packed Q/K/V projections on both templates.

    A backend failure in either template disables the experiment process-wide.
    The failing template first reconstructs its current dense weights, while
    subsequent layers take the ordinary dequant/copy branch in ``apply_pack``.
    """

    def __init__(self, on_disable=None):
        self.enabled = True
        self.reason = None
        self._on_disable = on_disable

    def disable(self, reason):
        if not self.enabled:
            return
        self.enabled = False
        self.reason = f"{type(reason).__name__}: {reason}"
        if self._on_disable is not None:
            self._on_disable(self.reason)


class DynamicPackedQ8QKV:
    """Persistent, template-owned row-int8 storage for KDA Q/K/V.

    ``apply_pack`` copies the current layer out of its recycled whole-layer
    staging buffers into these arenas. Patched ``nn.Linear.forward`` methods
    then consume stable views with ``aten::_weight_int8pack_mm``. Dense
    parameters remain available as a cold fallback; they are reconstructed
    from the owned packed weights before a runtime backend error is exposed to
    the caller.
    """

    _QBUF = "_k3_q8_qkv_weight_arena"
    _SBUF = "_k3_q8_qkv_scale_arena"
    _CTRL = "_k3_q8_qkv_controller"

    def __init__(self, module, dev, state, matmul, arenas=None):
        attention = getattr(module, "self_attn", None)
        if attention is None:
            raise ValueError("dynamic q8 Q/K/V requires module.self_attn")
        if getattr(attention, self._CTRL, None) is not None:
            raise ValueError("dynamic q8 Q/K/V is already installed")

        projections = {}
        q_cursor = 0
        scale_cursor = 0
        slots = {}
        for role in _QKV_ROLES:
            projection = getattr(attention, f"{role}_proj", None)
            if projection is None or not hasattr(projection, "weight"):
                raise ValueError(f"KDA attention has no {role}_proj.weight")
            if projection.bias is not None:
                raise ValueError(f"{role}_proj must be bias-free")
            shape = tuple(projection.weight.shape)
            if len(shape) != 2:
                raise ValueError(f"{role}_proj.weight must be a matrix")
            rows, cols = shape
            q_cursor = _align(q_cursor)
            scale_cursor = _align(scale_cursor * 4) // 4
            slots[role] = (q_cursor, rows * cols, scale_cursor, rows, cols)
            q_cursor += rows * cols
            scale_cursor += rows
            projections[role] = projection

        q_size = _align(q_cursor)
        scale_size = _align(scale_cursor * 4) // 4
        if arenas is None:
            q_arena = torch.empty(q_size, dtype=torch.int8, device=dev)
            scale_arena = torch.empty(
                scale_size,
                dtype=torch.float32,
                device=dev,
            )
        else:
            q_arena, scale_arena = arenas
            q_device_mismatch = (
                q_arena.device.type != dev.type
                or (
                    dev.index is not None
                    and q_arena.device.index != dev.index
                )
            )
            scale_device_mismatch = (
                scale_arena.device.type != dev.type
                or (
                    dev.index is not None
                    and scale_arena.device.index != dev.index
                )
            )
            if (q_arena.dtype != torch.int8 or q_arena.numel() < q_size
                    or q_device_mismatch):
                raise ValueError("shared packed Q/K/V weight arena is incompatible")
            if (scale_arena.dtype != torch.float32
                    or scale_arena.numel() < scale_size
                    or scale_device_mismatch):
                raise ValueError("shared packed Q/K/V scale arena is incompatible")
        attention.register_buffer(self._QBUF, q_arena, persistent=False)
        attention.register_buffer(self._SBUF, scale_arena, persistent=False)
        setattr(attention, self._CTRL, self)

        self.module = module
        self.attention = attention
        self.state = state
        self.matmul = matmul
        self.projections = projections
        self.slots = slots
        self.loaded = set()
        self.dense_loaded = set()
        self.packed_project_calls = 0
        self._prior_instance_forwards = {}
        self._original_forwards = {}
        for role, projection in projections.items():
            self._prior_instance_forwards[role] = projection.__dict__.get(
                "forward", None)
            self._original_forwards[role] = projection.forward

            def packed_forward(this_projection, hidden, _role=role,
                               _controller=self):
                return _controller.project(_role, hidden)

            projection.forward = types.MethodType(packed_forward, projection)

    @property
    def enabled(self):
        return self.state.enabled

    @property
    def nbytes(self):
        q_arena = getattr(self.attention, self._QBUF)
        scale_arena = getattr(self.attention, self._SBUF)
        return (
            q_arena.numel() * q_arena.element_size()
            + scale_arena.numel() * scale_arena.element_size()
        )

    def _views(self, role):
        qoff, qn, soff, rows, cols = self.slots[role]
        q_arena = getattr(self.attention, self._QBUF)
        scale_arena = getattr(self.attention, self._SBUF)
        return (
            q_arena[qoff:qoff + qn].view(rows, cols),
            scale_arena[soff:soff + rows],
        )

    def arenas(self):
        return (
            getattr(self.attention, self._QBUF),
            getattr(self.attention, self._SBUF),
        )

    @staticmethod
    def consumes(name):
        return name in _QKV_NAMES

    def begin_load(self):
        self.loaded.clear()
        self.dense_loaded.clear()

    def load(self, name, qweight, scale):
        """Copy one Q/K/V role into owned storage; return True if consumed."""
        if not self.enabled:
            return False
        role = _QKV_NAMES.get(name)
        if role is None:
            return False
        try:
            destination_q, destination_scale = self._views(role)
            if destination_q.shape != qweight.shape:
                raise ValueError(
                    f"packed {role} shape changed: "
                    f"{tuple(qweight.shape)} != {tuple(destination_q.shape)}")
            if scale.numel() != destination_scale.numel():
                raise ValueError(
                    f"packed {role} scale count changed: "
                    f"{scale.numel()} != {destination_scale.numel()}")
            # These are real copies, not aliases of _stage_buf(). That staging
            # storage is overwritten as soon as the next layer is applied.
            destination_q.copy_(qweight)
            destination_scale.copy_(scale)
        except (NotImplementedError, RuntimeError, TypeError, ValueError,
                MemoryError) as exc:
            # The caller still owns qweight/scale and will immediately run the
            # ordinary dense branch for this role. Preserve every earlier role
            # first, then make all remaining roles dense process-wide.
            self._materialize_dense()
            self.state.disable(exc)
            return False
        self.loaded.add(role)
        return True

    def mark_dense(self, name):
        """Record that apply_pack installed this layer's dense projection."""
        role = _QKV_NAMES.get(name)
        if role is not None:
            self.dense_loaded.add(role)

    def _materialize_dense(self, roles=None):
        selected = _QKV_ROLES if roles is None else roles
        for role in selected:
            if role not in self.loaded:
                continue
            qweight, scale = self._views(role)
            projection = self.projections[role]
            weight = projection.weight
            if (weight.device.type == "meta"
                    or tuple(weight.shape) != tuple(qweight.shape)
                    or weight.dtype != torch.float32):
                dense = qweight.to(torch.float32) * scale[:, None]
                projection.weight = torch.nn.Parameter(
                    dense, requires_grad=False)
            else:
                torch.mul(
                    qweight.to(torch.float32),
                    scale[:, None],
                    out=weight.data,
                )

    def finish_load(self):
        """Finish one layer, accepting all-packed or safely densifying hybrids."""
        missing = set(_QKV_ROLES) - self.loaded - self.dense_loaded
        if not missing:
            if self.enabled and self.dense_loaded:
                # A partial int8 spine is valid: apply_pack has already installed
                # the dense role(s). Rebuild the packed subset so this whole
                # current layer can take nn.Linear after the shared kill switch.
                self._materialize_dense()
                error = RuntimeError(
                    "packed KDA Q/K/V layer includes dense projection(s): "
                    + ", ".join(sorted(self.dense_loaded))
                )
                self.state.disable(error)
            return
        self._materialize_dense(self.loaded)
        error = RuntimeError(
            "KDA Q/K/V layer is incomplete; no packed or dense value for "
            + ", ".join(sorted(missing))
        )
        self.state.disable(error)
        # A role not recorded by either path may still contain a prior layer's
        # weight (or meta storage), so continuing would silently corrupt output.
        raise error

    def project(self, role, hidden):
        if not self.enabled or role not in self.loaded:
            return self._original_forwards[role](hidden)
        qweight, scale = self._views(role)
        try:
            flat = hidden.reshape(-1, hidden.shape[-1])
            output = self.matmul(flat, qweight, scale)
            output = output.view(*hidden.shape[:-1], qweight.shape[0])
            self.packed_project_calls += 1
            return output
        except (NotImplementedError, RuntimeError, TypeError) as exc:
            # The packed operator may be present in the dispatcher yet reject
            # a future MPS generation, shape, or PyTorch ABI. Rebuild all three
            # current weights before switching every template to dense mode.
            self._materialize_dense()
            self.state.disable(exc)
            return self._original_forwards[role](hidden)

    def uninstall(self):
        """Restore the original modules after a transactional setup failure."""
        for role, projection in self.projections.items():
            previous = self._prior_instance_forwards[role]
            if previous is None:
                projection.__dict__.pop("forward", None)
            else:
                projection.forward = previous
        if getattr(self.attention, self._CTRL, None) is self:
            delattr(self.attention, self._CTRL)
        for name in (self._QBUF, self._SBUF):
            if name in self.attention._buffers:
                del self.attention._buffers[name]


def install_dynamic_q8_qkv(module, dev, state, matmul, arenas=None):
    return DynamicPackedQ8QKV(module, dev, state, matmul, arenas=arenas)


def dynamic_q8_qkv(module):
    attention = getattr(module, "self_attn", None)
    return getattr(attention, DynamicPackedQ8QKV._CTRL, None)


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


# ---------------------------------------------------- chunked job planning ---
# Which packed buffer a job writes into.  The plan (paths, file offsets, byte
# counts, destination offsets) is identical on every token, so it is built once
# per layer and only the memoryview slices are materialised per call.
_QB, _SCB, _OB = 0, 1, 2


def plan_jobs(lay):
    """-> [(path, file_off, buf_id, dest_off, nbytes)] split at spine_io.CHUNK
    and sorted longest-first for a balanced greedy makespan."""
    jobs = lay.get("jobs")
    if jobs is not None:
        return jobs
    int8_dir = lay["int8_dir"]
    raw = []
    for _n, full, _sh, qoff, nb, scoff, rows in lay["items"]:
        base = os.path.join(int8_dir, full)
        raw.append((base + ".i8", _QB, qoff, nb))
        raw.append((base + ".sc", _SCB, scoff * 2, rows * 2))
    for _n, _f, _dt, _sh, ooff, nb, bp in lay["oplan"]:
        raw.append((bp, _OB, ooff, nb))
    ch = spine_io.CHUNK
    jobs = []
    for path, bid, doff, nb in raw:
        if ch <= 0 or nb <= ch:
            jobs.append((path, 0, bid, doff, nb))
            continue
        o = 0
        while o < nb:
            k = min(ch, nb - o)
            jobs.append((path, o, bid, doff + o, k))
            o += k
    jobs.sort(key=lambda j: -j[4])
    lay["jobs"] = jobs
    return jobs


def _rdadvise_next(prefix):
    """Kick the kernel's readahead for the layer AFTER the one we are about to
    read, so its pages are already in flight when the preloader gets there.
    Only fires once a layer has been seen before (its layout is cached), i.e.
    from the second forward pass on -- prefill is unaffected."""
    try:
        head, _, tail = prefix.rpartition("layers.")
        idx = int(tail.rstrip("."))
    except (ValueError, AttributeError):
        return
    nxt = _layouts.get(f"{head}layers.{idx + 1}.")
    if nxt is None:
        return
    seen = set()
    for path, _fo, _b, _d, _n in plan_jobs(nxt):
        if path not in seen:
            seen.add(path)
            spine_io.rdadvise(path)


# ------------------------------------------------------------- read side ----
def read_pack(module, prefix, int8_dir, res_dir, inv, spine, load_resident):
    """Runs in the preloader thread.  Returns a pack the main thread applies."""
    lay = plan_layout(module, prefix, int8_dir, res_dir, inv, spine)
    items = lay["items"]
    qbuf = _acquire(lay["qtotal"])
    scbuf = _acquire(lay["sctotal"] * 2)
    obuf = _acquire(max(lay["ototal"], 1))

    if spine_io.ENABLED:
        t0 = _time.time()
        if spine_io.RDADVISE:
            _rdadvise_next(prefix)
        bufs = (memoryview(qbuf), memoryview(scbuf), memoryview(obuf))
        jobs = [(path, fo, bufs[bid][doff:doff + nb])
                for path, fo, bid, doff, nb in plan_jobs(lay)]
        spine_io.run_jobs(jobs, _pool_exec(), READ_THREADS,
                          nocache=spine_io.stream_tier(prefix))
        PHASE["read_s"] += _time.time() - t0
        PHASE["read_bytes"] += lay["qtotal"] + lay["ototal"]
        return {"lay": lay, "q": qbuf, "sc": scbuf, "other": obuf}

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
    packed_qkv = dynamic_q8_qkv(module)
    if packed_qkv is not None:
        packed_qkv.begin_load()
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
        if packed_qkv is not None and packed_qkv.load(name, q, sc):
            continue
        p = module.get_parameter(name)
        fits = (p.device.type != "meta" and p.shape == torch.Size(shape)
                and p.dtype == torch.float32 and dt == torch.float32)
        if fits and use_metal:
            dequant_into(p.data, q, sc)
            if packed_qkv is not None and packed_qkv.consumes(name):
                packed_qkv.mark_dense(name)
            continue
        if fits and DEQ == "mulout":
            torch.mul(q, sc.view(rows, 1).to(torch.float32), out=p.data)
            if packed_qkv is not None and packed_qkv.consumes(name):
                packed_qkv.mark_dense(name)
            continue
        t = (q.to(torch.float32) * sc.view(rows, 1).to(torch.float32)).to(dt)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
        if packed_qkv is not None and packed_qkv.consumes(name):
            packed_qkv.mark_dense(name)
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
        if packed_qkv is not None and packed_qkv.consumes(name):
            packed_qkv.mark_dense(name)
    for name, full, path in lay["other"]:
        if full in have:
            continue
        t = load_resident(full).to(dev, dt)            # not on disk: rare fallback
        p = module.get_parameter(name)
        if p.device.type == "meta" or p.shape != t.shape:
            set_param(module, name, t)
        else:
            p.data.copy_(t)
        if packed_qkv is not None and packed_qkv.consumes(name):
            packed_qkv.mark_dense(name)
    if packed_qkv is not None:
        packed_qkv.finish_load()
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
    if spine_io.ENABLED:
        bits.append("| io: " + spine_io.describe())
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
