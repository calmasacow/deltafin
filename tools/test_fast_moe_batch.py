#!/usr/bin/env python3
"""Bit-exactness gate for tools/fast_moe_batch.py vs tools/fast_moe.py.

Uses real cached experts (both the .npz and the raw .bin shard-span layout). Correctness
only — deliberately tiny, no throughput measurement.

  python tools/test_fast_moe_batch.py [L1-E0 L1-E1 L1-E10 ...]
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fast_moe
import fast_moe_batch as fmb
from mxfp4 import dequant_mxfp4

ECACHE = os.path.join(ROOT, "k3-experts")

# raw .bin = verbatim shard span, 17,547,264 bytes:
#   w1_p 5505024 | w1_s 344064 | w2_p 5505024 | w2_s 344064 | w3_p 5505024 | w3_s 344064
BIN_SIZE = 17547264
_SPANS = [("w1", "p", 5505024, (3072, 1792)), ("w1", "s", 344064, (3072, 112)),
          ("w2", "p", 5505024, (3584, 1536)), ("w2", "s", 344064, (3584, 96)),
          ("w3", "p", 5505024, (3072, 1792)), ("w3", "s", 344064, (3072, 112))]


def load_expert(tag):
    npz, bin_ = os.path.join(ECACHE, tag + ".npz"), os.path.join(ECACHE, tag + ".bin")
    if os.path.exists(npz):
        z = np.load(npz)
        raw = {w: (np.ascontiguousarray(z[w + "_p"]), np.ascontiguousarray(z[w + "_s"]))
               for w in ("w1", "w2", "w3")}
        return raw, "npz"
    buf = np.fromfile(bin_, dtype=np.uint8)
    assert buf.size == BIN_SIZE, f"{tag}: {buf.size} != {BIN_SIZE}"
    raw, off = {}, 0
    for w, kind, n, shape in _SPANS:
        arr = np.ascontiguousarray(buf[off:off + n].reshape(shape))
        raw.setdefault(w, {})[kind] = arr
        off += n
    return {w: (d["p"], d["s"]) for w, d in raw.items()}, "bin"


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def relerr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(np.max(d / np.maximum(np.abs(b), 1e-30))), float(
        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    tags = sys.argv[1:] or ["L1-E0", "L1-E1", "L1-E10"]
    print(f"pool threads requested={fmb.THREADS}, created={fmb.pool_init()}")

    experts, srcs = {}, []
    for i, t in enumerate(tags):
        raw, src = load_expert(t)
        experts[i] = raw
        srcs.append(f"{t}({src})")
    ne = len(experts)
    print(f"experts: {', '.join(srcs)}")
    d_model = experts[0]["w1"][0].shape[1] * 2
    d_ff = experts[0]["w1"][0].shape[0]
    print(f"shapes: w1/w3 [{d_ff},{d_model}] w2 [{experts[0]['w2'][0].shape[0]},{d_ff}]")

    rng = np.random.default_rng(20260728)
    fails = 0

    # ---- 1. single GEMV: batched (n=1) vs old per-call mxfp4_gemv_mt --------------
    x = np.ascontiguousarray(rng.standard_normal(d_model, dtype=np.float32))
    p, s = experts[0]["w1"]
    y_old = fast_moe._gemv(p, s, x)
    y_new = np.empty(d_ff, dtype=np.float32)
    fmb.gemv_batch([(p, s, x, y_new)])
    ok = np.array_equal(bits(y_old), bits(y_new))
    print(f"[1] single GEMV batched vs fast_moe._gemv : {'BIT-EXACT' if ok else 'DIFFERS'}"
          f"  max_rel={relerr(y_new, y_old)[0]:.3e}")
    fails += not ok

    # fp64 oracle for absolute sanity (one matrix only)
    W = dequant_mxfp4(p, s).astype(np.float64)
    y64 = W @ x.astype(np.float64)
    del W
    print(f"    vs numpy fp64 dequant@x: max_rel={relerr(y_new, y64)[0]:.3e} "
          f"relL2={relerr(y_new, y64)[1]:.3e}")

    # ---- 2. batched multi-GEMV of a whole phase vs one-at-a-time ------------------
    mats, refs = [], []
    for e in range(ne):
        for w in ("w1", "w3"):
            pp, ss = experts[e][w]
            yy = np.empty(pp.shape[0], dtype=np.float32)
            mats.append((pp, ss, x, yy))
            refs.append(fast_moe._gemv(pp, ss, x))
    fmb.gemv_batch(mats)
    ok = all(np.array_equal(bits(m[3]), bits(r)) for m, r in zip(mats, refs))
    print(f"[2] {len(mats)}-matrix batch (shared x)      : {'BIT-EXACT' if ok else 'DIFFERS'}")
    fails += not ok

    # ---- 3. mxfp4_moe_layer (requested C signature) vs the same ------------------
    n = len(mats)
    pp = np.array([m[0].__array_interface__["data"][0] for m in mats], dtype=np.uintp)
    ss = np.array([m[1].__array_interface__["data"][0] for m in mats], dtype=np.uintp)
    rows = np.array([m[0].shape[0] for m in mats], dtype=np.int32)
    cols = np.array([m[0].shape[1] * 2 for m in mats], dtype=np.int32)
    out = np.zeros(int(rows.sum()), dtype=np.float32)
    fmb._LIB.mxfp4_moe_layer(pp, ss, rows, cols, n, x, out, fmb.THREADS)
    off, ok = 0, True
    for r, ref in zip(rows, refs):
        ok &= np.array_equal(bits(out[off:off + r]), bits(ref))
        off += int(r)
    print(f"[3] mxfp4_moe_layer concatenated output   : {'BIT-EXACT' if ok else 'DIFFERS'}")
    fails += not ok

    # ---- 4. expert_ffn: batched vs fast_moe --------------------------------------
    for e in range(ne):
        a = fast_moe.expert_ffn(experts[e], x)
        b = fmb.expert_ffn(experts[e], x)
        ok = np.array_equal(bits(a), bits(b))
        mr, l2 = relerr(b, a)
        print(f"[4] expert_ffn {tags[e]:<10} : {'BIT-EXACT' if ok else 'DIFFERS'} "
              f"max_rel={mr:.3e} relL2={l2:.3e}")
        fails += not ok

    # ---- 5. full moe_infer_fast, multi-token, all experts routed -----------------
    for N in (1, 3):
        xt = torch.from_numpy(np.ascontiguousarray(
            rng.standard_normal((N, d_model), dtype=np.float32)))
        ids = torch.tensor([list(range(ne))] * N, dtype=torch.long)
        wt = torch.from_numpy(rng.random((N, ne)).astype(np.float32))
        a = fast_moe.moe_infer_fast(xt, ids, wt, experts).numpy()
        b = fmb.moe_infer_fast(xt, ids, wt, experts).numpy()
        ok = np.array_equal(bits(a), bits(b))
        mr, l2 = relerr(b, a)
        print(f"[5] moe_infer_fast N={N} k={ne}          : {'BIT-EXACT' if ok else 'DIFFERS'} "
              f"max_rel={mr:.3e} relL2={l2:.3e}")
        fails += not ok
        # opt-in C-SiTU one-call variant (expected: close, not bit-exact)
        c = fmb.moe_infer_fused(xt, ids, wt, experts).numpy()
        mr, l2 = relerr(c, a)
        print(f"    moe_infer_fused (C SiTU, opt-in)     : "
              f"{'bit-exact' if np.array_equal(bits(a), bits(c)) else 'differs'} "
              f"max_rel={mr:.3e} relL2={l2:.3e}")

    # ---- 6. pool robustness: rebuild at different widths, repeat dispatches ------
    ref = fast_moe.expert_ffn(experts[0], x)
    for nt in (1, 2, 4, 6):
        fmb.pool_init(nt)
        got = fmb.expert_ffn(experts[0], x, nthreads=nt)
        ok = np.array_equal(bits(ref), bits(got))
        print(f"[6] nthreads={nt} (pool={fmb.pool_threads()})           : "
              f"{'BIT-EXACT' if ok else 'DIFFERS'}")
        fails += not ok
    fmb.pool_init(fmb.THREADS)
    for i in range(20):                      # repeated dispatch / idle-wake cycling
        got = fmb.expert_ffn(experts[0], x)
        if not np.array_equal(bits(ref), bits(got)):
            fails += 1
            print(f"[7] repeat dispatch {i} DIFFERS")
            break
    else:
        print("[7] 20 back-to-back dispatches           : BIT-EXACT (no pool drift)")

    # ---- 8. realistic top-16 layer shape (32-mat phase A + 16-mat phase B) -------
    # reuses the same 3 experts so no extra expert files are read; also exercises
    # duplicate pointers inside one batch.
    K = 16
    sel = [i % ne for i in range(K)]
    xt = torch.from_numpy(np.ascontiguousarray(
        rng.standard_normal((1, d_model), dtype=np.float32)))
    ids = torch.tensor([sel], dtype=torch.long)
    wt = torch.from_numpy(rng.random((1, K)).astype(np.float32))
    a = fast_moe.moe_infer_fast(xt, ids, wt, experts).numpy()
    b = fmb.moe_infer_fast(xt, ids, wt, experts).numpy()
    ok = np.array_equal(bits(a), bits(b))
    print(f"[8] top-{K} layer (48 GEMVs -> 2 dispatch): "
          f"{'BIT-EXACT' if ok else 'DIFFERS'} max_rel={relerr(b, a)[0]:.3e}")
    fails += not ok

    fmb.pool_shutdown()
    print("\nRESULT:", "ALL BIT-EXACT" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
