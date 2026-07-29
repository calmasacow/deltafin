#!/usr/bin/env python3
"""Resident-spine weight codecs — the one place the on-disk formats are defined.

numpy only (no torch), so the converter stays light. tools/mixed_spine.py
implements the read side on GPU and is validated against the `dequant_*`
oracles here.

CODECS
  i8   <name>.i8  int8   [rows, cols]        + <name>.sc fp16 [rows]   1.00 B/elt
       per-ROW absmax/127. Byte-identical to tools/convert_spine_int8.py, so an
       i8 tensor in a mixed spine is the same file the int8 spine already has.
  i6   <name>.i6  uint8  [rows, cw*3/4]      + <name>.s6 fp16 [rows,ng] 0.78 B/elt
       per-GROUP absmax/31, 4 six-bit values packed into 3 bytes.
  i4   <name>.i4  uint8  [rows, cw/2]        + <name>.s4 fp16 [rows,ng] 0.53 B/elt
       per-GROUP absmax/7, two nibbles per byte, LOW nibble = even column.
       Bit-identical to tools/convert_spine_int4.py's original output.

  cw = ceil(cols/g)*g  (the tail group is zero-padded); g must be a multiple of 8.

Group scales are rounded to fp16 BEFORE quantizing, so the dequant a loader
reconstructs is exactly what the converter measured.

MEASURED weight error on real spine tensors (tools/spine_sensitivity.py,
median relative Frobenius over 22 roles x 7 layer depths):
  i8 0.0093 | i6 g64 0.0245 (2.6x) | i4 g32 0.0975 (10.5x) | i4 g64 0.1083 (11.6x)
"""
import numpy as np

CODECS = ("i8", "i6", "i4")
BITS = {"i8": 8, "i6": 6, "i4": 4}
LEVELS = {"i8": 127, "i6": 31, "i4": 7}
EXT = {"i8": (".i8", ".sc"), "i6": (".i6", ".s6"), "i4": (".i4", ".s4")}


def n_groups(cols, g):
    return (cols + g - 1) // g


def padded_cols(cols, g):
    return n_groups(cols, g) * g


def blob_sizes(codec, rows, cols, g):
    """(payload bytes, scale bytes) for a [rows, cols] tensor."""
    if codec == "i8":
        return rows * cols, rows * 2
    cw = padded_cols(cols, g)
    ng = cw // g
    payload = rows * cw // 2 if codec == "i4" else rows * cw * 3 // 4
    return payload, rows * ng * 2


def bytes_per_elt(codec, cols, g):
    if codec == "i8":
        return 1.0 + 2.0 / cols
    return BITS[codec] / 8.0 + 2.0 / g


# ------------------------------------------------------------- quantize -----
def _pad(w, g):
    rows, cols = w.shape
    cw = padded_cols(cols, g)
    if cw == cols:
        return w, cw
    return np.concatenate([w, np.zeros((rows, cw - cols), np.float32)], axis=1), cw


def quantize(w, codec, g):
    """fp32 [rows, cols] -> (payload uint8 array, scales fp16 array, dequant fp32
    [rows, cols]).  The returned dequant is exactly what a loader reconstructs."""
    if codec == "i8":
        s = np.abs(w).max(axis=1, keepdims=True) / 127.0
        s[s == 0] = 1.0
        q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
        s16 = s.astype(np.float16).reshape(-1)
        # kimi_run dequantizes with the STORED fp16 scale
        deq = q.astype(np.float32) * s16.astype(np.float32)[:, None]
        return q.view(np.uint8), s16, deq
    L = float(LEVELS[codec])
    wp, cw = _pad(w, g)
    rows = wp.shape[0]
    ng = cw // g
    wg = wp.reshape(rows, ng, g)
    s16 = (np.abs(wg).max(axis=2) / L).astype(np.float16)
    s16[s16 == 0] = np.float16(6e-8)
    s = s16.astype(np.float32)[:, :, None]
    q = np.clip(np.rint(wg / s), -L, L).astype(np.int8)
    deq = (q.astype(np.float32) * s).reshape(rows, cw)[:, :w.shape[1]]
    u = q.reshape(rows, cw).view(np.uint8)
    if codec == "i4":
        n = u & np.uint8(0x0F)
        packed = (n[:, 0::2] | (n[:, 1::2] << np.uint8(4))).astype(np.uint8)
    else:                                   # i6: 4 values -> 3 bytes
        v = (u & np.uint8(0x3F)).reshape(rows, cw // 4, 4).astype(np.uint16)
        b0 = (v[:, :, 0] | (v[:, :, 1] << 6)) & 0xFF
        b1 = ((v[:, :, 1] >> 2) | (v[:, :, 2] << 4)) & 0xFF
        b2 = ((v[:, :, 2] >> 4) | (v[:, :, 3] << 2)) & 0xFF
        packed = np.stack([b0, b1, b2], axis=2).reshape(rows, cw * 3 // 4).astype(np.uint8)
    return packed, s16, deq


# ------------------------------------------------------------- dequantize ---
def dequant(codec, payload, scales, rows, cols, g):
    """numpy oracle: raw blobs -> fp32 [rows, cols]. mixed_spine's GPU kernels
    are checked bit-for-bit against this."""
    q = np.frombuffer(payload, dtype=np.uint8)
    s = np.frombuffer(scales, dtype=np.float16).astype(np.float32)
    if codec == "i8":
        return q.view(np.int8).reshape(rows, cols).astype(np.float32) * s.reshape(rows, 1)
    cw = padded_cols(cols, g)
    ng = cw // g
    if codec == "i4":
        b = q.reshape(rows, cw // 2)
        lo = (b & 0x0F).astype(np.int16)
        hi = (b >> 4).astype(np.int16)
        lo = np.where(lo > 7, lo - 16, lo)
        hi = np.where(hi > 7, hi - 16, hi)
        v = np.stack([lo, hi], axis=2).reshape(rows, cw)
    else:
        b = q.reshape(rows, cw // 4, 3).astype(np.int16)
        v0 = b[:, :, 0] & 0x3F
        v1 = (b[:, :, 0] >> 6) | ((b[:, :, 1] & 0x0F) << 2)
        v2 = (b[:, :, 1] >> 4) | ((b[:, :, 2] & 0x03) << 4)
        v3 = b[:, :, 2] >> 2
        v = np.stack([v0, v1, v2, v3], axis=2).reshape(rows, cw)
        v = np.where(v > 31, v - 64, v)
    return (v.astype(np.float32) * s.reshape(rows, ng)[:, :, None].repeat(g, axis=1)
            .reshape(rows, cw))[:, :cols]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w = rng.standard_normal((37, 200), dtype=np.float32) * 0.02
    w[3, 5] = 0.4
    w[10] = 0.0
    for codec in CODECS:
        for g in (32, 64, 128):
            p, s, deq = quantize(w, codec, g)
            back = dequant(codec, p.tobytes(), s.tobytes(), 37, 200, g)
            assert np.array_equal(back, deq), (codec, g, np.abs(back - deq).max())
            e = np.linalg.norm(deq - w) / np.linalg.norm(w)
            pb, sb = blob_sizes(codec, 37, 200, g)
            assert p.nbytes == pb and s.nbytes == sb, (codec, g, p.nbytes, pb, s.nbytes, sb)
            print(f"{codec} g={g:3d}  rel_fro={e:.4f}  {bytes_per_elt(codec,200,g):.3f} B/elt"
                  f"  blobs {pb}/{sb} OK")
            if codec == "i8":
                break
    print("round-trip OK: dequant(quantize(w)) is bit-identical to the "
          "converter's own dequant, for every codec")
