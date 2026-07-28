#!/usr/bin/env python3
"""Correctness tests for the int4 spine converter + loader. Small and fast:
synthetic round-trips plus, if an int4 blob exists, a bit-exactness check of the
loader against the numpy oracle on real spine data. No benchmarking."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import convert_spine_int4 as C
import int4_loader as L

FAIL = 0


def check(cond, msg):
    global FAIL
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAIL += 1


def t_pack_roundtrip():
    """quantize_block -> raw blobs -> loader.dequant must be bit-exact."""
    print("[1] pack/unpack round-trip (synthetic)")
    rng = np.random.default_rng(1)
    for (rows, cols, g) in [(7, 128, 64), (4, 256, 32), (3, 384, 128), (5, 64, 64)]:
        w = rng.standard_normal((rows, cols), dtype=np.float32) * 0.02
        w[0, :] = 0.0                                   # all-zero row
        w[1, 3] = 5.0                                   # outlier
        packed, s16, deq = C.quantize_block(w, g)
        got = L.dequant(packed.tobytes(), s16.tobytes(), (rows, cols), group=g,
                        dev="cpu", dtype=torch.float32).numpy()
        check(np.array_equal(got, deq), f"[{rows},{cols}] g={g} loader == converter dequant")
        check(np.array_equal(deq, L.quantize_reference(w, g)),
              f"[{rows},{cols}] g={g} converter == numpy oracle")
        lvl = np.unique(np.rint(deq[2] / s16[2].astype(np.float32).repeat(g)[:cols]))
        check(lvl.min() >= -7 and lvl.max() <= 7, f"g={g} levels within [-7,7]")


def t_padding():
    """cols not a multiple of g: tail group zero-padded, loader unpads."""
    print("[2] ragged tail group (cols % g != 0)")
    rng = np.random.default_rng(2)
    rows, cols, g = 3, 100, 64                          # ng=2, cw=128
    w = rng.standard_normal((rows, cols), dtype=np.float32)
    cw = C.n_groups(cols, g) * g
    wp = np.concatenate([w, np.zeros((rows, cw - cols), np.float32)], axis=1)
    packed, s16, deq = C.quantize_block(wp, g)
    check(packed.shape == (rows, cw // 2) and s16.shape == (rows, cw // g), "blob shapes")
    got = L.dequant(packed.tobytes(), s16.tobytes(), (rows, cols), group=g).numpy()
    check(got.shape == (rows, cols), "loader returns unpadded shape")
    check(np.array_equal(got, deq[:, :cols]), "unpadded values bit-exact")


def t_sizes():
    print("[3] declared blob sizes")
    for (r, c, g) in [(12288, 7168, 64), (896, 7168, 32), (163840, 7168, 128), (3, 100, 64)]:
        eb, sb = C.blob_sizes(r, c, g)
        ng = C.n_groups(c, g)
        check(eb == r * ng * g // 2 and sb == r * ng * 2, f"[{r},{c}] g={g} -> {eb}B + {sb}B")


def t_error_scale():
    """int4-g64 error must sit between int8 and int4-per-row, and shrink with g."""
    print("[4] error monotonicity in g")
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((512, 1024), dtype=np.float32) * 0.02).astype(np.float32)
    e = {}
    for g in (32, 64, 128, 256):
        d = L.quantize_reference(w, g)
        e[g] = float(np.linalg.norm(d - w) / np.linalg.norm(w))
    print("     rel_fro " + "  ".join(f"g{g}={v:.3e}" for g, v in e.items()))
    check(e[32] < e[64] < e[128] < e[256], "error strictly increases with group size")


def t_device():
    print("[5] device parity (mps vs cpu)")
    if not torch.backends.mps.is_available():
        print("  skip  no MPS"); return
    rng = np.random.default_rng(4)
    w = rng.standard_normal((64, 512), dtype=np.float32) * 0.01
    packed, s16, deq = C.quantize_block(w, 64)
    a = L.dequant(packed.tobytes(), s16.tobytes(), (64, 512), group=64, dev="cpu")
    b = L.dequant(packed.tobytes(), s16.tobytes(), (64, 512), group=64, dev="mps").cpu()
    check(torch.equal(a, b), "mps dequant bit-identical to cpu")
    c = L.dequant(packed.tobytes(), s16.tobytes(), (64, 512), group=64,
                  dev="mps", dtype=torch.float16).cpu().float()
    check(float((c - a).abs().max()) < 1e-3, "fp16 dequant within fp16 rounding")


def t_chunking():
    print("[6] row chunking gives the same result as one shot")
    rng = np.random.default_rng(5)
    w = rng.standard_normal((129, 256), dtype=np.float32) * 0.05
    packed, s16, _ = C.quantize_block(w, 64)
    a = L.dequant(packed.tobytes(), s16.tobytes(), (129, 256), group=64)
    b = L.dequant(packed.tobytes(), s16.tobytes(), (129, 256), group=64, chunk_elems=1024)
    check(torch.equal(a, b), "chunked == unchunked")


def t_real():
    """If the converter has produced any blob, verify it against the bf16 source."""
    print("[7] real spine tensor, converter -> disk -> loader vs bf16 oracle")
    if not L.available():
        print("  skip  no int4 blobs yet (run convert_spine_int4.py --names ...)")
        return
    names = sorted(f[:-3] for f in os.listdir(L.INT4_DIR) if f.endswith(".i4"))
    g = L.group_size()
    for name in names[:2]:
        rows, cols = L.shape_of(name)
        nr = max(1, min(rows, (32 << 20) // (cols * 2)))
        raw = np.fromfile(os.path.join(C.RES, name), dtype=np.uint16,
                          count=nr * cols).reshape(nr, cols)
        w = (raw.astype(np.uint32) << 16).view(np.float32)
        ref = L.quantize_reference(w, g)
        q, s = L.read_blobs(name)
        cw = C.n_groups(cols, g) * g
        got = L.dequant(q[:nr * cw // 2], s[:nr * (cw // g) * 2], (nr, cols),
                        group=g).numpy()
        rel = float(np.linalg.norm(ref - w) / np.linalg.norm(w))
        check(np.array_equal(got, ref),
              f"{name.split('layers.')[-1]} first {nr} rows bit-exact (rel_fro={rel:.3e})")


for fn in (t_pack_roundtrip, t_padding, t_sizes, t_error_scale, t_device, t_chunking, t_real):
    fn()
print("\nFAILURES:", FAIL)
sys.exit(1 if FAIL else 0)
