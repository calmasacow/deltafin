#!/usr/bin/env python3
"""Quality gate for the int4 resident spine: weight-space AND output-space error
of int4-g{32,64,128} against bf16, on the SAME tensors as int8, so the two are
directly comparable.

Reads a handful of real spine tensors READ-ONLY from k3-resident/tensors, one
row-slice each (default 48 MB — a contiguous prefix of rows; relative Frobenius
error is a per-row statistic, so a prefix is representative and keeps the disk
quiet). Nothing is written to the spine; nothing is converted.

Metrics per tensor and per scheme:
  rel_fro   ||Wq - W||_F / ||W||_F                    weight-space
  rel_l2    ||Wq x - W x||_2 / ||W x||_2, x ~ N(0,I)  output-space (what a GEMV
                                                      actually sees; same
                                                      statistic as the Metal
                                                      MXFP4 validation)
Schemes:
  int8      per-row absmax/127, as convert_spine_int8.py ships it
            (scale computed in fp32, stored fp16 — the loader then uses the
             fp16 value, so the shipped path carries a small extra error)
  int8x     the same but scale rounded to fp16 before quantizing (what the
            converter should do; shown to separate quantizer error from the
            scale-rounding bug)
  int4-gG   group-wise absmax/7 over G consecutive columns, fp16 scale rounded
            before use — exactly tools/convert_spine_int4.py / int4_loader.py

  python tools/int4_quality_report.py                 # default 7 tensors
  python tools/int4_quality_report.py --max-mb 16     # even lighter
  python tools/int4_quality_report.py --json out.json
"""
import argparse, json, os, sys, time
import numpy as np

K3 = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(K3, "k3-resident/tensors")
INV = json.load(open(os.path.join(K3, "k3-meta/tensor_inventory_offsets.json")))
P = "language_model.model.layers."

# A representative spread: attention projections (KDA + MLA), the MoE router
# gate, shared-expert MLP, the fused routed-expert projections, and lm_head
# (the one tensor whose error lands directly on the logits / top-5 order).
DEFAULT = [
    P + "0.self_attn.q_proj.weight",        # KDA layer
    P + "3.self_attn.o_proj.weight",        # MLA layer
    P + "3.self_attn.kv_b_proj.weight",
    P + "3.block_sparse_moe.gate.weight",
    P + "3.block_sparse_moe.shared_experts.down_proj.weight",
    P + "30.block_sparse_moe.routed_expert_up_proj.weight",
    "language_model.lm_head.weight",
]


def read_rows(name, max_bytes):
    """Contiguous row prefix of a bf16 spine tensor as fp32. READ ONLY."""
    rows, cols = INV[name]["shape"]
    nr = max(1, min(rows, max_bytes // (cols * 2)))
    raw = np.fromfile(os.path.join(RES, name), dtype=np.uint16,
                      count=nr * cols).reshape(nr, cols)
    return (raw.astype(np.uint32) << 16).view(np.float32), rows, cols, nr


def deq_int8(w, round_scale_first):
    a = np.abs(w).max(axis=1, keepdims=True)
    if round_scale_first:
        s = (a / 127.0).astype(np.float16).astype(np.float32)
    else:
        s = a / 127.0                                    # as convert_spine_int8.py
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    st = (s.astype(np.float16).astype(np.float32) if not round_scale_first else s)
    return q.astype(np.float32) * st                     # loader uses the fp16 scale


def deq_int4(w, g):
    rows, cols = w.shape
    ng = (cols + g - 1) // g
    cw = ng * g
    if cw != cols:
        w = np.concatenate([w, np.zeros((rows, cw - cols), np.float32)], axis=1)
    wg = w.reshape(rows, ng, g)
    s16 = (np.abs(wg).max(axis=2) / 7.0).astype(np.float16)
    s16[s16 == 0] = np.float16(6e-8)
    s = s16.astype(np.float32)[:, :, None]
    q = np.clip(np.rint(wg / s), -7, 7).astype(np.int8)
    return (q.astype(np.float32) * s).reshape(rows, cw)[:, :cols]


def sym_alpha(w, g, alpha, levels=7):
    """Symmetric int4 with the group absmax shrunk by `alpha` (MSE-clip search)."""
    rows, cols = w.shape
    ng = cols // g
    wg = w.reshape(rows, ng, g)
    s = (np.abs(wg).max(2) * alpha / levels).astype(np.float16).astype(np.float32)[:, :, None]
    s[s == 0] = 1e-8
    return (np.clip(np.rint(wg / s), -levels, levels) * s).reshape(rows, cols)


def asym16(w, g):
    """Asymmetric int4: 16 levels between the group min and max, fp16 scale AND
    fp16 zero-point -> 4 bytes per group instead of 2 (0.5 + 4/g B/elt)."""
    rows, cols = w.shape
    ng = cols // g
    wg = w.reshape(rows, ng, g)
    lo, hi = wg.min(2)[:, :, None], wg.max(2)[:, :, None]
    s = ((hi - lo) / 15.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1e-8
    z = lo.astype(np.float16).astype(np.float32)
    return (np.clip(np.rint((wg - z) / s), 0, 15) * s + z).reshape(rows, cols)


def sweep(name, max_bytes):
    """Is the plain absmax quantizer leaving anything on the table at 4 bits?
    (Answer, measured: no — it is already at the Lloyd-Max scalar floor.)"""
    w, rows, cols, nr = read_rows(name, max_bytes)
    nw = np.linalg.norm(w)
    r = lambda d: float(np.linalg.norm(d - w) / nw)
    print(f"\n=== 4-bit quantizer variants on {name} (rows {nr}/{rows}) ===")
    print(f"    Lloyd-Max 16-level Gaussian floor: rel_fro = {10**(-20.22/20):.4e}")
    for g in (32, 64, 128):
        base = r(sym_alpha(w, g, 1.0))
        best = min((r(sym_alpha(w, g, a)), a) for a in (0.80, 0.85, 0.90, 0.95, 1.00))
        a16 = r(asym16(w, g))
        print(f"    g={g:3d}  absmax={base:.4e} ({0.5+2/g:.4f} B/elt)   "
              f"best-clip a={best[1]:.2f} -> {best[0]:.4e} ({100*(1-best[0]/base):+.1f}%)   "
              f"asym16={a16:.4e} ({0.5+4/g:.4f} B/elt)")


def bytes_per_elt(scheme, cols):
    if scheme.startswith("int8"):
        return 1.0 + 2.0 / cols
    g = int(scheme.split("-g")[1])
    return 0.5 + 2.0 / g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=int, default=48,
                    help="MB of each tensor to read (row prefix), default 48")
    ap.add_argument("--tensors", nargs="+", default=DEFAULT)
    ap.add_argument("--probes", type=int, default=8, help="random x vectors for rel_l2")
    ap.add_argument("--json", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep clip factor / asymmetric int4 on one tensor")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    schemes = ["int8", "int8x", "int4-g32", "int4-g64", "int4-g128"]
    rows_out, t0 = [], time.time()

    for name in args.tensors:
        if name not in INV or not os.path.exists(os.path.join(RES, name)):
            print(f"  SKIP (absent) {name}", flush=True)
            continue
        w, rows, cols, nr = read_rows(name, args.max_mb << 20)
        x = rng.standard_normal((cols, args.probes), dtype=np.float32)
        y = w @ x
        ny = np.linalg.norm(y)
        nw = np.linalg.norm(w)
        rec = {"name": name, "shape": [rows, cols], "rows_used": nr}
        for s in schemes:
            d = deq_int8(w, s == "int8x") if s.startswith("int8") else deq_int4(w, int(s.split("-g")[1]))
            e = d - w
            rec[s] = {"rel_fro": float(np.linalg.norm(e) / (nw + 1e-30)),
                      "rel_l2": float(np.linalg.norm(d @ x - y) / (ny + 1e-30)),
                      "max_abs": float(np.abs(e).max()),
                      "bytes_per_elt": bytes_per_elt(s, cols)}
            del d, e
        rows_out.append(rec)
        short = name.replace("language_model.model.layers.", "L").replace("language_model.", "")
        print(f"{short}  [{rows},{cols}] rows_used={nr}", flush=True)
        for s in schemes:
            r = rec[s]
            print(f"    {s:10s} rel_fro={r['rel_fro']:.4e}  rel_l2={r['rel_l2']:.4e}"
                  f"  x{r['rel_fro']/rec['int8']['rel_fro']:.2f} vs int8", flush=True)
        del w, x, y

    print("\n=== summary (size-weighted over full tensors) ===")
    wts = np.array([r["shape"][0] * r["shape"][1] for r in rows_out], float)
    print(f"{'scheme':10s} {'rel_fro med':>12s} {'rel_fro wt':>12s} {'rel_l2 med':>12s} "
          f"{'rel_l2 wt':>12s} {'x int8':>8s} {'spine GB':>9s}")
    base = None
    for s in schemes:
        f = np.array([r[s]["rel_fro"] for r in rows_out])
        l = np.array([r[s]["rel_l2"] for r in rows_out])
        bpe = np.mean([r[s]["bytes_per_elt"] for r in rows_out])
        fw = float((f * wts).sum() / wts.sum())
        if base is None:
            base = fw
        print(f"{s:10s} {np.median(f):12.4e} {fw:12.4e} {np.median(l):12.4e} "
              f"{float((l*wts).sum()/wts.sum()):12.4e} {fw/base:8.2f} "
              f"{113.456 * bpe / 2:9.1f}")
    print(f"\n(bf16 spine = 113.5 GB; read {sum(r['rows_used']*r['shape'][1]*2 for r in rows_out)/1e6:.0f} MB "
          f"in {time.time()-t0:.0f}s)")
    if args.sweep and rows_out:
        sweep(rows_out[0]["name"], args.max_mb << 20)
    if args.json:
        json.dump(rows_out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
