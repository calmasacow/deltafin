#!/usr/bin/env python3
"""Expert-locality study over k3-meta/router_trace.jsonl (+ router_trace.run1.jsonl).

Decides whether a RAM-pinned hot-expert tier is worth building. Pure analysis, no model.

Trace shape: one JSON record per (forward pass, MoE layer) with a FLAT ids list that is
the concatenation of top-16 selections for every token in the batch (80 ids = 5 prefill
tokens). Records appear in execution order; a new forward pass starts whenever the layer
index stops increasing. The file is an append log across many runs, so it contains
repeated prompts -- every measurement below is therefore reported twice: once over the
raw log and once over a de-duplicated "unique context" view that keeps only the first
occurrence of each distinct layer-1 routing signature.

  python tools/expert_locality.py
"""
import json, os, sys
from collections import Counter, defaultdict, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERT_BYTES = 17_547_264
TOPK = 16
N_EXPERTS = 896
CACHE_GB = (5, 10, 20, 30, 40)
TOPNS = (16, 64, 128, 256, 448)
TRAIN_FRAC = 0.60


def load_passes(paths):
    recs = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_src"] = os.path.basename(p)
                recs.append(r)
    passes, cur, prev = [], [], 1 << 30
    for r in recs:
        if r["layer"] <= prev:
            if cur:
                passes.append(cur)
            cur = []
        cur.append(r)
        prev = r["layer"]
    if cur:
        passes.append(cur)
    return recs, passes


def to_token_passes(passes):
    """-> list of {layer: (e0..e15)}, one per (forward pass, token), in execution order."""
    out = []
    for p in passes:
        ntok = len(p[0]["ids"]) // TOPK
        assert all(len(r["ids"]) % TOPK == 0 for r in p)
        tp = [dict() for _ in range(ntok)]
        for r in p:
            ids = r["ids"]
            for t in range(min(ntok, len(ids) // TOPK)):
                tp[t][r["layer"]] = tuple(ids[t * TOPK:(t + 1) * TOPK])
        out.extend(tp)
    return out


def dedup(tps):
    """Keep first occurrence of each distinct layer-1 signature (proxy for token+context)."""
    seen, out = set(), []
    for tp in tps:
        k = tp.get(1) or tp.get(min(tp)) if tp else None
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append(tp)
    return out


def sel_counts(tps):
    """(layer,expert) -> selection count, and layer -> Counter."""
    glob, per_layer = Counter(), defaultdict(Counter)
    for tp in tps:
        for L, ids in tp.items():
            per_layer[L].update(ids)
            for e in ids:
                glob[(L, e)] += 1
    return glob, per_layer


def skew(per_layer, tag):
    tot_layers = sorted(per_layer)
    print(f"\n  top-N cumulative selection mass, averaged over {len(tot_layers)} MoE layers  [{tag}]")
    print(f"    {'N':>5} {'of 896':>8} {'mass':>8}   {'worst layer':>12} {'best layer':>11}")
    for n in TOPNS:
        fr = []
        for L in tot_layers:
            c = per_layer[L]
            t = sum(c.values())
            fr.append(sum(v for _, v in c.most_common(n)) / t)
        print(f"    {n:>5} {n/N_EXPERTS*100:>7.1f}% {sum(fr)/len(fr)*100:>7.1f}%   "
              f"{min(fr)*100:>11.1f}% {max(fr)*100:>10.1f}%")
    # uniform reference
    print(f"    (uniform routing would give N/896: "
          + ", ".join(f"top-{n}={n/N_EXPERTS*100:.1f}%" for n in TOPNS) + ")")


def n_slots(gb):
    return int(gb * 1e9 // EXPERT_BYTES)


def pinned_hitrate(train_tps, eval_tps, tag):
    train_c, _ = sel_counts(train_tps)
    eval_c, _ = sel_counts(eval_tps)
    n_eval = sum(eval_c.values())
    if not n_eval:
        return
    bytes_tok = 92 * TOPK * EXPERT_BYTES / 1e9
    print(f"\n  frequency-pinned RAM tier  [{tag}]  "
          f"train={len(train_tps)} token-passes, eval={len(eval_tps)} ({n_eval} selections)")
    print(f"    {'size':>6} {'slots':>6} {'holdout':>9} {'in-sample':>10} "
          f"{'LRU':>7}   {'disk GB/token after holdout tier':>32}")
    for gb in CACHE_GB:
        k = n_slots(gb)
        hot = set(x for x, _ in train_c.most_common(k))
        hold = sum(v for x, v in eval_c.items() if x in hot) / n_eval
        orac = sum(v for _, v in eval_c.most_common(k)) / n_eval
        lru = lru_hitrate(train_tps, eval_tps, k)
        per_layer = Counter(L for L, _ in hot)
        print(f"    {gb:>4} GB {k:>6} {hold*100:>8.1f}% {orac*100:>9.1f}% "
              f"{lru*100:>6.1f}%   {bytes_tok*(1-hold):>27.2f} GB"
              f"   [{k/92:.1f} experts/layer avg, "
              f"{min(per_layer.values())}..{max(per_layer.values())} range]")
    print(f"    (full model = {92*N_EXPERTS*EXPERT_BYTES/1e9:.0f} GB of experts; "
          f"cold cost = {bytes_tok:.2f} GB/token)")


def lru_hitrate(train_tps, eval_tps, cap):
    """LRU over (layer,expert) slots, warmed on train, measured on eval."""
    od = OrderedDict()
    def touch(k):
        hit = k in od
        if hit:
            od.move_to_end(k)
        else:
            od[k] = 1
            if len(od) > cap:
                od.popitem(last=False)
        return hit
    for tp in train_tps:
        for L, ids in tp.items():
            for e in ids:
                touch((L, e))
    hits = tot = 0
    for tp in eval_tps:
        for L, ids in tp.items():
            for e in ids:
                hits += touch((L, e))
                tot += 1
    return hits / max(tot, 1)


def coactivation(train_tps, eval_tps, tag):
    """P(expert e' at layer L+1 | expert e at layer L), and whether it beats static
    frequency as a prefetch predictor for the next layer's set."""
    pair = defaultdict(Counter)       # (L, e) -> Counter(e' at L+1)
    src = Counter()                   # (L, e) -> times seen
    nxt_freq = defaultdict(Counter)   # L+1 -> Counter(e')
    for tp in train_tps:
        for L, ids in tp.items():
            nx = tp.get(L + 1)
            if nx is None:
                continue
            nxt_freq[L + 1].update(nx)
            for e in ids:
                src[(L, e)] += 1
                pair[(L, e)].update(nx)

    # --- conditional probability / lift of the single best successor ---
    lifts, top1 = [], []
    for key, c in pair.items():
        L, _ = key
        n = src[key]
        if n < 20:
            continue
        marg = nxt_freq[L + 1]
        tot = sum(marg.values()) / TOPK        # number of (token-pass) observations at L+1
        e2, cnt = c.most_common(1)[0]
        p = cnt / n
        base = marg[e2] / tot if tot else 0
        top1.append(p)
        if base > 0:
            lifts.append(p / base)
    if top1:
        top1.sort(); lifts.sort()
        print(f"\n  cross-layer co-activation  [{tag}]  "
              f"{len(top1)} (layer,expert) sources with >=20 observations")
        print(f"    P(best successor at L+1 | e at L): median {top1[len(top1)//2]*100:.1f}%  "
              f"p90 {top1[int(len(top1)*0.9)]*100:.1f}%  max {top1[-1]*100:.1f}%")
        print(f"    lift over that successor's own marginal: median {lifts[len(lifts)//2]:.2f}x  "
              f"p90 {lifts[int(len(lifts)*0.9)]:.2f}x  max {lifts[-1]:.2f}x")
        print(f"    (marginal P(specific expert in a top-16 set) = 16/896 = 1.8%)")

    # --- practical test: predict S_{L+1} from S_L, vs static frequency ---
    def score_next(L, ids):
        sc = Counter()
        for e in ids:
            c = pair.get((L, e))
            if c:
                n_e = src[(L, e)]
                for e2, v in c.items():
                    sc[e2] += v / n_e
        return sc

    print(f"\n    next-layer prediction recall on holdout  [{tag}]")
    print(f"    {'M':>5} {'co-activation':>14} {'static-freq':>12} {'same-idx':>9}"
          f"   (recall of the 16 experts actually used at L+1)")
    for M in (16, 32, 64, 128):
        co_r, st_r, pv_r, n = 0.0, 0.0, 0.0, 0
        for tp in eval_tps:
            for L, ids in tp.items():
                nx = tp.get(L + 1)
                if nx is None or (L + 1) not in nxt_freq:
                    continue
                pred_co = set(x for x, _ in score_next(L, ids).most_common(M))
                pred_st = set(x for x, _ in nxt_freq[L + 1].most_common(M))
                nxs = set(nx)
                co_r += len(pred_co & nxs) / len(nxs)
                st_r += len(pred_st & nxs) / len(nxs)
                pv_r += len(nxs & set(ids)) / len(nxs)   # "expert i at L also at L+1"
                n += 1
        if n:
            print(f"    {M:>5} {co_r/n*100:>13.1f}% {st_r/n*100:>11.1f}% {pv_r/n*100:>8.1f}%"
                  f"   (random = {M/N_EXPERTS*100:.1f}%)")

    # --- byte economics: prefetching only pays if it fetches fewer bytes than demand ---
    # Baseline per layer: fetch the |A \ R| experts that are missing from the pinned tier.
    # With prefetch: fetch the M predicted non-resident experts one layer early, then
    # demand-fetch whatever the prediction missed.
    train_c, _ = sel_counts(train_tps)
    print(f"\n    prefetch byte economics on holdout  [{tag}] "
          f"(experts fetched per layer, 16 needed)")
    print(f"    {'tier':>7} {'M':>5} {'demand-only':>12} {'prefetch+demand':>16} {'verdict':>9}")
    for gb in (0, 20, 40):
        hot = set(x for x, _ in train_c.most_common(n_slots(gb))) if gb else set()
        for M in (8, 16, 32):
            base = pf = n = 0
            for tp in eval_tps:
                for L, ids in tp.items():
                    nx = tp.get(L + 1)
                    if nx is None or (L + 1) not in nxt_freq:
                        continue
                    A = set(nx)
                    miss = set(e for e in A if (L + 1, e) not in hot)
                    pred = [x for x, _ in score_next(L, ids).most_common(M + 40)
                            if (L + 1, x) not in hot][:M]
                    ps = set(pred)
                    base += len(miss)
                    pf += len(ps) + len(miss - ps)
                    n += 1
            if n:
                b, p = base / n, pf / n
                print(f"    {(str(gb)+' GB') if gb else 'none':>7} {M:>5} {b:>12.2f} "
                      f"{p:>16.2f} {('WIN' if p < b else 'loss'):>9}")


def prev_token(tps, tag):
    """Same-layer overlap between consecutive token-passes (prefetch from previous token)."""
    ov = []
    for a, b in zip(tps, tps[1:]):
        for L in set(a) & set(b):
            ov.append(len(set(a[L]) & set(b[L])) / TOPK)
    if ov:
        print(f"\n  previous-token same-layer expert recall [{tag}]: "
              f"{sum(ov)/len(ov)*100:.1f}%  over {len(ov)} layer pairs "
              f"(colibri GLM published baseline: 41.3%)")


def main():
    paths = [os.path.join(ROOT, "k3-meta/router_trace.jsonl"),
             os.path.join(ROOT, "k3-meta/router_trace.run1.jsonl")]
    recs, passes = load_passes(paths)
    tps = to_token_passes(passes)
    uniq = dedup(tps)
    layer_hits = Counter(r["layer"] for r in recs)
    n_sel = sum(len(r["ids"]) for r in recs)
    complete = sum(1 for p in passes if len(p) == 92)

    print("=" * 78)
    print("TRACE INVENTORY")
    print("=" * 78)
    print(f"  records {len(recs)}  forward passes {len(passes)} "
          f"({complete} complete 92-layer, rest aborted/partial runs)")
    print(f"  token-passes {len(tps)}  ->  {len(uniq)} distinct layer-1 signatures "
          f"({len(uniq)/len(tps)*100:.0f}% unique; the log is an append across many "
          f"re-runs of a few prompts)")
    print(f"  total routing selections {n_sel}  MoE layers {min(layer_hits)}..{max(layer_hits)}")
    uniq_pairs = len(set((L, e) for tp in tps for L, ids in tp.items() for e in ids))
    print(f"  distinct (layer,expert) pairs touched {uniq_pairs} "
          f"of {92*N_EXPERTS} = {uniq_pairs/(92*N_EXPERTS)*100:.1f}%  "
          f"({uniq_pairs*EXPERT_BYTES/1e9:.1f} GB)")
    print(f"  layer coverage: {sum(1 for L in layer_hits if layer_hits[L] >= 50)} "
          f"of 92 layers have >=50 records")

    for tag, data in (("raw log", tps), ("unique contexts", uniq)):
        print("\n" + "=" * 78)
        print(f"PART 1 - SELECTION SKEW  [{tag}]")
        print("=" * 78)
        _, per_layer = sel_counts(data)
        skew(per_layer, tag)

    print("\n" + "=" * 78)
    print("PART 2 - FREQUENCY-PINNED RAM TIER (60/40 chronological holdout)")
    print("=" * 78)
    for tag, data in (("raw log", tps), ("unique contexts", uniq)):
        cut = int(len(data) * TRAIN_FRAC)
        pinned_hitrate(data[:cut], data[cut:], tag)
        prev_token(data, tag)

    print("\n" + "=" * 78)
    print("PART 3 - CROSS-LAYER CO-ACTIVATION")
    print("=" * 78)
    for tag, data in (("raw log", tps), ("unique contexts", uniq)):
        cut = int(len(data) * TRAIN_FRAC)
        coactivation(data[:cut], data[cut:], tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
