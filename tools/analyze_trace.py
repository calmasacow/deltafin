#!/usr/bin/env python3
"""Analyze router_trace.jsonl from lazy-K3 runs: the first real K3 routing data.

Reports the numbers the optimization plan hinges on:
 - per-layer expert-selection frequency skew (top-N cumulative routing mass)
 - temporal locality: expert overlap between consecutive decode steps
 - previous-token routing recall (colibri PILOT baseline comparison)
 - simulated cache hit-rate vs cache size (LRU and frequency-pinned)
"""
import json, os, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "k3-meta/router_trace.jsonl")

by_layer = defaultdict(Counter)          # layer -> Counter(expert -> count)
by_step = defaultdict(dict)              # step -> {layer: [ids]}
for line in open(path):
    r = json.loads(line)
    ids = r["ids"]
    by_layer[r["layer"]].update(ids)
    by_step[r["step"]][r["layer"]] = ids

steps = sorted(by_step)
layers = sorted(by_layer)
total_sel = sum(sum(c.values()) for c in by_layer.values())
print(f"trace: {len(steps)} steps x {len(layers)} MoE layers, {total_sel} selections")

# --- skew: cumulative mass of top-N experts per layer, averaged ---
for topn in (16, 64, 128, 256, 448):
    fracs = []
    for li in layers:
        c = by_layer[li]
        tot = sum(c.values())
        fracs.append(sum(n for _, n in c.most_common(topn)) / tot)
    print(f"avg cumulative routing mass, top-{topn:3d}/896 experts: {sum(fracs)/len(fracs)*100:.1f}%")

# --- temporal locality between consecutive decode steps ---
if len(steps) >= 3:
    overlaps, prev_recall = [], []
    for a, b in zip(steps[1:], steps[2:]):  # decode steps only (skip prefill step 0)
        for li in set(by_step[a]) & set(by_step[b]):
            sa, sb = set(by_step[a][li]), set(by_step[b][li])
            if sb:
                overlaps.append(len(sa & sb) / len(sb))
    if overlaps:
        print(f"consecutive-token expert overlap (prev-token routing recall): "
              f"{sum(overlaps)/len(overlaps)*100:.1f}%  (colibri GLM baseline: 41.3%)")

# --- unique experts touched so far (cache growth) ---
uniq = {(li, e) for li, c in by_layer.items() for e in c}
gb = len(uniq) * 17.5e-3
print(f"unique (layer,expert) pairs touched: {len(uniq)} (~{gb:.1f} GB at MXFP4)")

# --- simulated hit rate vs frequency-pinned cache size ---
allc = Counter()
for li, c in by_layer.items():
    for e, n in c.items():
        allc[(li, e)] += n
for cache_gb in (10, 17, 25, 40):
    n_slots = int(cache_gb * 1e3 / 17.5)
    hot = set(k for k, _ in allc.most_common(n_slots))
    hits = sum(n for k, n in allc.items() if k in hot)
    print(f"frequency-pinned cache {cache_gb:2d} GB ({n_slots} experts): "
          f"hit rate {hits/total_sel*100:.1f}% (upper bound, self-trace)")
