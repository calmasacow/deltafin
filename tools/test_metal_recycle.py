"""Proof that the Metal MoE wrap cache survives fetch_v2's buffer recycling.

tools/metal_moe.py caches MTLBuffer wraps in the dylib keyed by BASE POINTER.
tools/fetch_v2.py's pread reader owns a pool of recyclable EXPERT_SPAN slots, so

    (a) the same base pointer carries a different expert a few layers later, and
    (b) a surplus slot is unmapped outright, freeing that address for reuse.

(a) is safe by aliasing: the wrap points at the live pages, so the GPU reads what
the last pread wrote.  (b) is the dangerous one — a wrap over unmapped memory,
silently reading dead pages if that address comes back as a different mapping.
metal_moe holds a strong reference to the owning mapping for as long as a wrap
exists and drops the wrap before releasing it, so (b) cannot happen; _sync_wrap
additionally drops a wrap whose owner object changed.

This test forces both conditions HARD and then checks the GPU against the CPU
kernel on every iteration:

    K3_PREAD_MAX_FREE=0   every slot is dropped the moment it is retired
    K3_PREAD_DEPTH=1      retirement happens after a single read_layer()
    K3_METAL_PIN_MAX=n    pins evict every iteration, so metal_moe actually lets
                          go of mappings and addresses really do get recycled

    ./venv/bin/python tools/test_metal_recycle.py
    ./venv/bin/python tools/test_metal_recycle.py --iters 8 --experts 8

--unsafe disables the invalidation (K3_METAL_STALE_OK=1) to show the check has
teeth.  It reads through wraps whose mapping may be gone: expect wrong numbers or
a crash.  Never used by the default run.
"""
import argparse
import collections
import os
import re
import sys

# these must be set before fetch_v2 / metal_moe read them at import time
_DEFAULTS = {
    "K3_EXPERT_READ": "pread",
    "K3_PREAD_MAX_FREE": "0",
    "K3_PREAD_DEPTH": "1",
    "K3_METAL_PIN_MAX": "8",
    "K3_PREAD_WORKERS": "6",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_moe                                             # noqa: E402
import fetch_v2                                             # noqa: E402
import metal_moe                                            # noqa: E402

ECACHE = fetch_v2.ECACHE
_BIN = re.compile(r"^L(\d+)-E(\d+)\.bin$")


def bin_index():
    """{layer: [eid, ...]} for every full-size .bin span in the cache."""
    by_layer = collections.defaultdict(list)
    with os.scandir(ECACHE) as it:
        for de in it:
            m = _BIN.match(de.name)
            if m and de.stat().st_size == fetch_v2.EXPERT_SPAN:
                by_layer[int(m.group(1))].append(int(m.group(2)))
    return {k: sorted(v) for k, v in by_layer.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6, help="layers to walk")
    ap.add_argument("--experts", type=int, default=8, help="experts per layer")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--unsafe", action="store_true",
                    help="disable wrap invalidation to show the test has teeth")
    args = ap.parse_args()
    if args.unsafe:
        os.environ["K3_METAL_STALE_OK"] = "1"
        metal_moe.STALE_OK = True

    if not metal_moe.available():
        raise SystemExit(f"metal unavailable: {metal_moe.last_error()}")
    if fetch_v2.EXPERT_READ != "pread":
        raise SystemExit("K3_EXPERT_READ must be pread for this test")

    idx = bin_index()
    layers = [L for L in sorted(idx) if len(idx[L]) >= args.experts][:args.iters]
    if len(layers) < 2:
        raise SystemExit("need >=2 layers with enough .bin experts in k3-experts/")

    print(f"metal ok  mode={'bindless' if metal_moe.stats()['bindless'] else 'per-expert'}  "
          f"pin_max={metal_moe.PIN_MAX}  pread_max_free={fetch_v2.PREAD_MAX_FREE}  "
          f"pread_depth={fetch_v2.PREAD_DEPTH}  unsafe={args.unsafe}")
    print(f"walking layers {layers} x {args.experts} experts")

    rng = np.random.default_rng(7)
    seen = collections.defaultdict(set)        # base ptr -> {(layer, eid)}
    owners = collections.defaultdict(set)      # base ptr -> {id(owner)}
    worst = 0.0
    for L in layers:
        eids = idx[L][:args.experts]
        raw = fetch_v2.fetch_experts(L, eids, dequant=False)
        # every span must be a page-aligned view into a pread slot, or the whole
        # zero-copy premise (and this test) is moot
        for e in eids:
            p, own = metal_moe._span_ptr(raw[e], 0)
            seen[p].add((L, e))
            owners[p].add(id(own))

        x = torch.from_numpy(
            (rng.standard_normal((1, metal_moe.HIDDEN)) * 0.5).astype(np.float32))
        w = rng.random(len(eids)).astype(np.float32)
        w /= w.sum()
        ids = torch.tensor([eids], dtype=torch.long)
        wt = torch.from_numpy(w[None, :])

        gpu = metal_moe.moe_infer_metal(x, ids, wt, raw).numpy()[0]
        cpu = fast_moe.moe_infer_fast(x, ids, wt, raw).numpy()[0]
        den = max(float(np.abs(cpu).max()), 1e-30)
        rel = float(np.abs(gpu - cpu).max()) / den
        worst = max(worst, rel)
        st = metal_moe.stats()
        print(f"  L{L:<3d} rel {rel:.3e}   wraps {st['zero_copy_wraps']:4d} "
              f"copies {st['copies']:3d} cache {st['cache_entries']:3d} "
              f"pins {st['pinned']:3d} rewraps {st['rewraps']:3d}")
        del raw                                # let fetch_v2 recycle/unmap freely

    reused = {p: v for p, v in seen.items() if len(v) > 1}
    multi_owner = {p: v for p, v in owners.items() if len(v) > 1}
    total = sum(len(v) for v in seen.values())
    print(f"\ndistinct base pointers {len(seen)} for {total} expert loads")
    print(f"pointers that carried >1 expert: {len(reused)}"
          + (f"  e.g. {hex(next(iter(reused)))} -> {sorted(next(iter(reused.values())))[:4]}"
             if reused else ""))
    print(f"pointers that changed owning mapping: {len(multi_owner)}")
    print(f"stale wraps dropped by _sync_wrap: {metal_moe._rewraps}")

    print(f"worst rel {worst:.3e}")
    if worst >= args.tol:
        print("FAIL", f"(tol {args.tol:g})")
        return 1
    if not reused:
        # e.g. K3_METAL_PIN_MAX high enough that every mapping stayed pinned, so
        # fetch_v2 could never hand an address back out. Correct, but it proves
        # nothing about recycling.
        print("INCONCLUSIVE: no pointer reuse observed — lower K3_METAL_PIN_MAX "
              "or raise --iters to force recycling")
        return 2
    print("PASS", f"(tol {args.tol:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
