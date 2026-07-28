#!/usr/bin/env python3
"""int4 resident-spine loader — the read side of tools/convert_spine_int4.py.

Mirrors kimi_run.py's _load_int8() so wiring it in is a small edit:

    import int4_loader
    t = int4_loader.load_int4(full, dev=DEV, dtype=DT)   # None when not converted

and, for the double-buffered preloader (worker thread reads bytes, main thread
builds tensors), the same split kimi_run.py already uses for ("i8", q, sc):

    rec = ("i4",) + int4_loader.read_blobs(full)         # in the worker
    t   = int4_loader.dequant(rec[1], rec[2], shape, dev=DEV, dtype=DT)

Format (see k3-resident-int4/meta.json):
  .i4  uint8 [rows, cw/2], cw = ceil(cols/g)*g, LOW nibble = even column
  .s4  float16 [rows, cw/g], one symmetric scale per g consecutive columns
  nibbles are 2's-complement 4-bit in [-7,7]; dequant = q * scale.

Unpacking happens on the target device (only cols/2 bytes cross the bus, which
is the entire point), with a CPU fallback if a backend lacks the bit ops.
This is OPT-IN: nothing here changes behaviour unless a caller asks for it.
"""
import json, os
import numpy as np
import torch

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT4_DIR = os.environ.get("K3_INT4_DIR") or os.path.join(ROOT, "k3-resident-int4/tensors")
_META_PATH = os.path.join(os.path.dirname(INT4_DIR.rstrip("/")), "meta.json")

_INV = None
_META = None
_BITOPS_OK = {}          # device type -> bool


def meta():
    global _META
    if _META is None:
        try:
            _META = json.load(open(_META_PATH))
        except FileNotFoundError:
            _META = {"format": "int4-sym-group", "version": 1, "group": 64,
                     "missing_meta": True}
    return _META


def group_size():
    return int(meta().get("group", 64))


def _inv():
    global _INV
    if _INV is None:
        _INV = json.load(open(os.path.join(ROOT, "k3-meta/tensor_inventory_offsets.json")))
    return _INV


def shape_of(name):
    return tuple(_inv()[name]["shape"])


def available():
    """True when an int4 spine has been built (any .i4 present)."""
    try:
        return any(f.endswith(".i4") for f in os.listdir(INT4_DIR))
    except OSError:
        return False


def has(name):
    return os.path.exists(os.path.join(INT4_DIR, name + ".i4"))


def read_blobs(name):
    """(packed_bytes, scale_bytes) — pure file I/O, safe to call off-thread."""
    with open(os.path.join(INT4_DIR, name + ".i4"), "rb") as f:
        q = f.read()
    with open(os.path.join(INT4_DIR, name + ".s4"), "rb") as f:
        s = f.read()
    return q, s


def _bitops(t):
    """(low_nibble, high_nibble) as float tensors, with a CPU fallback."""
    dev = t.device.type
    if _BITOPS_OK.get(dev, True):
        try:
            return t.bitwise_and(0x0F), t.bitwise_right_shift(4)
        except Exception:                        # backend without uint8 bit ops
            _BITOPS_OK[dev] = False
    n = t.to(torch.int16)
    return (n % 16), (n // 16)


def dequant(qbuf, sbuf, shape, group=None, dev="cpu", dtype=torch.float32,
            chunk_elems=64 << 20):
    """Reconstruct a [rows, cols] tensor on `dev` from the two raw blobs."""
    rows, cols = int(shape[0]), int(shape[1])
    g = int(group or group_size())
    ng = (cols + g - 1) // g
    cw = ng * g
    exp_q, exp_s = rows * cw // 2, rows * ng * 2
    if len(qbuf) != exp_q or len(sbuf) != exp_s:
        raise ValueError(f"int4 blob size mismatch for [{rows},{cols}] g={g}: "
                         f"got {len(qbuf)}/{len(sbuf)}, want {exp_q}/{exp_s}")
    dev = torch.device(dev)
    qcpu = torch.frombuffer(bytearray(qbuf), dtype=torch.uint8).reshape(rows, cw // 2)
    scpu = torch.frombuffer(bytearray(sbuf), dtype=torch.float16).reshape(rows, ng)
    out = torch.empty((rows, cols), dtype=dtype, device=dev)
    step = max(1, min(rows, int(chunk_elems) // max(cw, 1)))
    for r0 in range(0, rows, step):
        r1 = min(r0 + step, rows)
        qd = qcpu[r0:r1].to(dev, non_blocking=True)
        lo, hi = _bitops(qd)
        lo = lo.to(dtype)
        hi = hi.to(dtype)
        lo = torch.where(lo > 7, lo - 16, lo)     # 2's-complement 4-bit
        hi = torch.where(hi > 7, hi - 16, hi)
        w = torch.stack((lo, hi), dim=2).reshape(r1 - r0, cw)
        del lo, hi, qd
        w = w.view(r1 - r0, ng, g)
        w.mul_(scpu[r0:r1].to(dev, non_blocking=True).to(dtype).unsqueeze(-1))
        out[r0:r1] = w.view(r1 - r0, cw)[:, :cols]
        del w
    return out


def load_int4(name, dev="cpu", dtype=torch.float32, shape=None):
    """Dequantized [rows, cols] tensor on `dev`, or None if `name` isn't in the
    int4 spine. Drop-in shape/dtype match for kimi_run.py's _load_int8()."""
    if not has(name):
        return None
    q, s = read_blobs(name)
    return dequant(q, s, shape or shape_of(name), dev=dev, dtype=dtype)


def quantize_reference(w_f32, g):
    """numpy oracle: fp32 [rows, cols] -> dequantized fp32, exactly what the
    converter writes and this loader reads back. Used by the quality report and
    by any test that wants int4 error without touching disk."""
    rows, cols = w_f32.shape
    ng = (cols + g - 1) // g
    cw = ng * g
    if cw != cols:
        w_f32 = np.concatenate([w_f32, np.zeros((rows, cw - cols), np.float32)], axis=1)
    wg = w_f32.reshape(rows, ng, g)
    s16 = (np.abs(wg).max(axis=2) / 7.0).astype(np.float16)
    s16[s16 == 0] = np.float16(6e-8)
    s = s16.astype(np.float32)[:, :, None]
    q = np.clip(np.rint(wg / s), -7, 7).astype(np.int8)
    return (q.astype(np.float32) * s).reshape(rows, cw)[:, :cols]


if __name__ == "__main__":
    import sys
    if not available():
        print(f"no int4 spine at {INT4_DIR} — build one with "
              f"tools/convert_spine_int4.py --sample 4")
        sys.exit(1)
    print(f"[int4] {INT4_DIR}  meta={meta()}")
    names = sys.argv[1:] or sorted(f[:-3] for f in os.listdir(INT4_DIR)
                                   if f.endswith(".i4"))[:3]
    for n in names:
        t = load_int4(n, dev="cpu", dtype=torch.float32)
        print(f"  {n}  {tuple(t.shape)}  absmax={t.abs().max().item():.4f}  "
              f"nonzero={(t != 0).float().mean().item():.3f}")
