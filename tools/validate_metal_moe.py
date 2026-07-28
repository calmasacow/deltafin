"""Correctness gate for the Metal MoE bridge: tools/metal_moe.py vs tools/fast_moe.py
on real cached experts.  Deliberately tiny — a few experts, one token, no timing.

    ./venv/bin/python tools/validate_metal_moe.py [--layer 3] [--experts 0,1,2]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_moe          # noqa: E402  (CPU oracle, imported read-only)
import fetch_v2          # noqa: E402  (cache reader, imported read-only)
import metal_moe         # noqa: E402


def load(layer, eids):
    raw = {}
    for e in eids:
        r = fetch_v2._cache_load(layer, e)      # local cache only, never fetches
        if r is None:
            raise SystemExit(f"L{layer}-E{e} not in k3-experts/")
        raw[e] = r
    return raw


def report(tag, a, b):
    d = np.abs(a - b)
    scale = max(np.abs(b).max(), 1e-30)
    rel_max = d.max() / scale                                  # vs peak magnitude
    rel_l2 = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
    nz = np.abs(b) > 1e-3 * scale
    rel_ew = (d[nz] / np.abs(b[nz])).max() if nz.any() else 0.0
    print(f"  {tag:22s} relmax {rel_max:.3e}   relL2 {rel_l2:.3e}   "
          f"elemwise(>1e-3 peak) {rel_ew:.3e}")
    return rel_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", default="0,1,2")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    eids = [int(s) for s in args.experts.split(",")]

    if not metal_moe.available():
        raise SystemExit(f"metal unavailable: {metal_moe.last_error()}")
    print(f"metal ok  mode={'bindless' if metal_moe.stats()['bindless'] else 'per-expert'}")

    raw = load(args.layer, eids)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, metal_moe.HIDDEN), dtype=np.float32) * 0.5)
    w = rng.random(len(eids)).astype(np.float32)
    w /= w.sum()
    ids = torch.tensor([eids], dtype=torch.long)
    wt = torch.from_numpy(w[None, :])

    cpu = fast_moe.moe_infer_fast(x, ids, wt, raw).numpy()[0]
    print(f"L{args.layer} experts {eids}  |out|max {np.abs(cpu).max():.4f}")

    worst = 0.0
    for bindless in (1, 0):
        got = metal_moe.set_mode(bindless)
        if got != bindless:
            print(f"  mode {bindless} unsupported, skipped")
            continue
        gpu = metal_moe.moe_infer_metal(x, ids, wt, raw).numpy()[0]
        worst = max(worst, report("bindless" if bindless else "per-expert", gpu, cpu))
    print(" ", metal_moe.stats())
    print("PASS" if worst < args.tol else "FAIL", f"(tol {args.tol:g})")
    return 0 if worst < args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
