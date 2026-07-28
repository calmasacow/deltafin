#!/usr/bin/env python3
"""Gate for K3_EXPERT_READ=pread (tools/fetch_v2.py).

Proves the pread reader returns byte-identical expert tensors to the mmap path,
with the same dict structure, for the .bin cache, the legacy .npz cache, and a
mixed layer — and that the speculative prefetch path (K3_EXPERT_PREFETCH) does
not change a single byte either.

    python tools/test_expert_read.py [--layers 1 45 92]
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                              # noqa: E402
import fetch_v2                                                 # noqa: E402

SPAN = fetch_v2.EXPERT_SPAN
SHAPES = {"w1": ((3072, 1792), (3072, 112)),
          "w2": ((3584, 1536), (3584, 96)),
          "w3": ((3072, 1792), (3072, 112))}


def structure_ok(raw, eids):
    assert set(raw) == set(eids), f"expert set {sorted(set(raw))} != {sorted(eids)}"
    for e, ws in raw.items():
        assert set(ws) == {"w1", "w2", "w3"}, f"E{e} keys {sorted(ws)}"
        for w, (p, s) in ws.items():
            ep, es = SHAPES[w]
            assert isinstance(p, np.ndarray) and isinstance(s, np.ndarray)
            assert p.dtype == s.dtype == np.uint8, f"E{e} {w} dtype"
            assert p.shape == ep and s.shape == es, f"E{e} {w} shape {p.shape}/{s.shape}"
            assert p.flags["C_CONTIGUOUS"] and s.flags["C_CONTIGUOUS"], f"E{e} {w} strides"


def copy_of(raw):
    return {e: {w: (p.copy(), s.copy()) for w, (p, s) in ws.items()}
            for e, ws in raw.items()}


def compare(a, b, tag):
    assert set(a) == set(b), f"{tag}: expert sets differ"
    for e in a:
        for w in ("w1", "w2", "w3"):
            for i, part in enumerate(("packed", "scale")):
                x, y = a[e][w][i], b[e][w][i]
                if not np.array_equal(x, y):
                    n = int((x != y).sum())
                    raise AssertionError(f"{tag}: E{e} {w} {part} differs in {n} bytes")
    print(f"  {tag}: BIT-EXACT over {len(a)} experts "
          f"({len(a) * SPAN / 1e6:.0f} MB)")


def pick(layer, n, kind):
    """n expert ids in `layer` whose cache entry is .bin / .npz / either."""
    out = []
    for e in random.sample(range(896), 896):
        has_bin = fetch_v2._has_bin(layer, e)
        has_npz = os.path.exists(fetch_v2._cache_path(layer, e))
        if kind == "bin" and not has_bin:
            continue
        if kind == "npz" and (has_bin or not has_npz):
            continue
        if kind == "any" and not (has_bin or has_npz):
            continue
        out.append(e)
        if len(out) == n:
            break
    return sorted(out)


def run_case(layer, eids, tag):
    fetch_v2.EXPERT_READ = "mmap"
    ref = copy_of(fetch_v2.fetch_experts(layer, eids, dequant=False))
    structure_ok(ref, eids)
    fetch_v2.EXPERT_READ = "pread"
    t0 = time.time()
    got = fetch_v2.fetch_experts(layer, eids, dequant=False)
    dt = time.time() - t0
    structure_ok(got, eids)
    compare(ref, got, tag)
    print(f"      pread {dt * 1000:6.1f} ms for {len(eids)} experts "
          f"({len(eids) * SPAN / 1e9 / max(dt, 1e-9):.2f} GB/s)")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=[1, 45, 92])
    ap.add_argument("--seed", type=int, default=20260728)
    a = ap.parse_args()
    random.seed(a.seed)
    saved = fetch_v2.EXPERT_READ
    try:
        for L in a.layers:
            print(f"layer {L}")
            b = pick(L, 16, "bin")
            if b:
                run_case(L, b, f"L{L} 16x .bin")
            z = pick(L, 4, "npz")
            if z:
                run_case(L, z, f"L{L} 4x legacy .npz")
            if b and z:
                run_case(L, sorted(b[:12] + z), f"L{L} mixed .bin+.npz")

        # --- buffer recycling: two consecutive layers must not alias ---------
        L0, L1 = a.layers[0], a.layers[-1]
        e0, e1 = pick(L0, 16, "bin"), pick(L1, 16, "bin")
        fetch_v2.EXPERT_READ = "mmap"
        ref0 = copy_of(fetch_v2.fetch_experts(L0, e0, dequant=False))
        fetch_v2.EXPERT_READ = "pread"
        got0 = fetch_v2.fetch_experts(L0, e0, dequant=False)
        fetch_v2.fetch_experts(L1, e1, dequant=False)      # recycles nothing yet
        compare(ref0, got0, f"L{L0} still valid one layer later (DEPTH="
                            f"{fetch_v2.PREAD_DEPTH})")

        # --- speculative prefetch must not change any byte -------------------
        fetch_v2.drop_prefetch()
        spec = sorted(set(e1[:10] + pick(L1, 6, "bin")))    # partial overlap
        fetch_v2.prefetch_layer(L1, spec)
        got1 = fetch_v2.fetch_experts(L1, e1, dequant=False)
        fetch_v2.EXPERT_READ = "mmap"
        ref1 = copy_of(fetch_v2.fetch_experts(L1, e1, dequant=False))
        compare(ref1, got1, f"L{L1} after partial-overlap prefetch")
        st = fetch_v2.stats
        print(f"  prefetch hits {st['prefetch_hits']} wasted {st['prefetch_wasted']}")
        print(f"  slots allocated {st['pread_slots']} "
              f"({st['pread_slots'] * SPAN / 1e6:.0f} MB), "
              f"pread {st['pread_bytes'] / 1e9:.2f} GB")
    finally:
        fetch_v2.EXPERT_READ = saved
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
