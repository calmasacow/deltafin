#!/usr/bin/env python3
"""Verdict for an ab_pilot.sh run directory.

Statistics note. An earlier ad-hoc analysis of this comparison cross-paired
every run of one arm against every run of the other, then took a standard
error over the resulting deltas. That is pseudo-replication: N runs produced
N^2 "samples" that share run-level state, and the error bars came out several
times too tight. Here each iteration contributes exactly ONE independent
number -- the mean per-chunk delta between the two arms run adjacently within
that iteration -- and all inference happens across iterations.
"""
import glob
import os
import re
import statistics
import sys

STATS = re.compile(r"\[stats\] generated=(\d+) elapsed=([\d.]+)s")
GATE = re.compile(r"\[pilot-gate\] mode=\S+ threshold=(\S+) warmup=(\d+) passes=(\d+) "
                  r"experts issued=(\d+) suppressed=(\d+) prev-token-plans=(\d+)")
# Student t, two-sided 95%, by degrees of freedom.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
       8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def t_critical(df):
    if df <= 0:
        return float("nan")
    for key in sorted(T95):
        if df <= key:
            return T95[key]
    return 1.96


def read_manifest(outdir):
    path = os.path.join(outdir, "manifest.tsv")
    rows = []
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
    return rows


def chunk_series(log_path):
    """Cumulative [stats] lines -> (token-boundary tuple, per-chunk seconds)."""
    rows = []
    with open(log_path, errors="replace") as handle:
        for line in handle:
            found = STATS.search(line)
            if found:
                rows.append((int(found[1]), float(found[2])))
    boundaries = tuple(generated for generated, _ in rows)
    deltas = [(b[1] - a[1], b[0] - a[0]) for a, b in zip(rows, rows[1:])]
    return boundaries, deltas


def gate_summary(log_path):
    with open(log_path, errors="replace") as handle:
        for line in handle:
            found = GATE.search(line)
            if found:
                return (f"threshold={found[1]} warmup={found[2]} issued={found[4]} "
                        f"suppressed={found[5]} prev-token-plans={found[6]}")
    return None


def main(outdir, skip=4):
    rows = [r for r in read_manifest(outdir) if r["iteration"] != "0"]
    if not rows:
        print("no timed iterations recorded yet")
        return 0
    arms = sorted({r["arm"] for r in rows})
    if len(arms) != 2:
        print(f"expected exactly 2 arms, found: {arms}")
        return 1
    print(f"=== PILOT A/B: {os.path.basename(outdir.rstrip('/'))} ===")
    print(f"arms: {arms[0]} vs {arms[1]}   timed runs: {len(rows)}")

    # --- correctness: every run must generate identical text ---------------
    shas = {r["sha"] for r in rows}
    failed = [r for r in rows if r["exit"] != "0"]
    print("\nCORRECTNESS")
    if len(shas) == 1:
        print(f"  output identity  PASS  all {len(rows)} runs identical (sha {shas.pop()})")
    else:
        print(f"  output identity  *** FAIL ***  {len(shas)} distinct outputs: {shas}")
        print("  Speculative prefetch may never change what the router selects.")
        print("  Timing below is NOT valid until this is explained.")
    if failed:
        print(f"  nonzero exits    *** {len(failed)} run(s) failed ***")

    # --- per-arm whole-run summary -----------------------------------------
    print("\nPER-ARM (internal elapsed, whole run)")
    per_arm = {}
    for arm in arms:
        totals = [float(r["internal_s"]) for r in rows if r["arm"] == arm and r["internal_s"]]
        tokens = [int(r["tokens"]) for r in rows if r["arm"] == arm and r["tokens"]]
        per_arm[arm] = totals
        if totals:
            per_token = statistics.median(totals) / max(1, statistics.median(tokens))
            print(f"  {arm:10} n={len(totals)}  median {statistics.median(totals):8.1f}s"
                  f"  ({per_token:5.2f}s/token)  range [{min(totals):.1f}, {max(totals):.1f}]")

    # --- paired, one independent sample per iteration ----------------------
    # A short run can have fewer chunks than the warmup skip. Shrink the skip
    # rather than silently producing an empty section, and say so.
    shortest = min((len(chunk_series(os.path.join(outdir, r["log"]))[1])
                    for r in rows if os.path.exists(os.path.join(outdir, r["log"]))),
                   default=0)
    effective_skip = skip
    if shortest <= skip + 1:
        effective_skip = max(0, shortest // 3)
        print(f"\nNOTE: runs have only {shortest} chunks, fewer than the {skip}-chunk warmup"
              f" skip; using skip={effective_skip} instead. Timing from runs this short is"
              f" dominated by prefill and startup -- use -t 200 or more for a real verdict.")
    print(f"\nPAIRED PER-CHUNK (within iteration, first {effective_skip} chunks skipped for warmup)")
    skip = effective_skip
    iterations = sorted({int(r["iteration"]) for r in rows})
    per_iteration = []
    tokens_per_chunk = []
    for iteration in iterations:
        series = {}
        for row in rows:
            if int(row["iteration"]) == iteration:
                path = os.path.join(outdir, row["log"])
                if os.path.exists(path):
                    series[row["arm"]] = chunk_series(path)
        if len(series) != 2:
            continue
        (b0, d0), (b1, d1) = series[arms[0]], series[arms[1]]
        if b0 != b1 or len(d0) != len(d1):
            print(f"  iteration {iteration}: chunk boundaries differ between arms, skipped")
            continue
        pairs = [(y[0] - x[0], x[1]) for x, y in zip(d0[skip:], d1[skip:])]
        if not pairs:
            print(f"  iteration {iteration}: no chunks left after the warmup skip, excluded")
            continue
        delta = statistics.mean(p[0] for p in pairs)
        counts = [p[1] for p in pairs if p[1] > 0]
        per_iteration.append(delta)
        tokens_per_chunk.extend(counts)
        print(f"  iteration {iteration}: {arms[1]} - {arms[0]} = {delta:+.3f} s/chunk"
              f"  ({len(pairs)} chunks)")

    if len(per_iteration) >= 2:
        mean = statistics.mean(per_iteration)
        stdev = statistics.stdev(per_iteration)
        n = len(per_iteration)
        se = stdev / n ** 0.5
        df = n - 1
        crit = t_critical(df)
        lo, hi = mean - crit * se, mean + crit * se
        tpc = statistics.median(tokens_per_chunk) if tokens_per_chunk else 1
        print(f"\n  mean {mean:+.3f} s/chunk   se {se:.3f}   n={n} iterations (df={df})")
        print(f"  95% CI [{lo:+.3f}, {hi:+.3f}] s/chunk"
              f"  = [{lo / tpc:+.3f}, {hi / tpc:+.3f}] s/token at {tpc:.0f} tokens/chunk")
        significant = (lo > 0) or (hi < 0)
        if significant:
            direction = "SLOWER" if mean > 0 else "FASTER"
            print(f"  VERDICT: {arms[1]} is {direction} than {arms[0]} by "
                  f"{abs(mean / tpc):.3f} s/token (95% CI excludes zero)")
        else:
            resolvable = crit * se / tpc
            print(f"  VERDICT: no significant difference. This experiment could only "
                  f"resolve effects larger than ~{resolvable:.3f} s/token;")
            print(f"           a smaller real difference cannot be excluded by this data.")
    elif per_iteration:
        print(f"\n  only {len(per_iteration)} iteration: {per_iteration[0]:+.3f} s/chunk, "
              f"no error bar. Run more iterations (-n) before drawing a conclusion.")

    # --- order effect: does position 1 differ from position 2? -------------
    print("\nORDER EFFECT (guards against a systematic first/second advantage)")
    for position in ("1", "2"):
        totals = [float(r["internal_s"]) for r in rows
                  if r["position"] == position and r["internal_s"]]
        if totals:
            print(f"  position {position}: n={len(totals)} median {statistics.median(totals):8.1f}s")
    flips = {}
    for row in rows:
        if row["position"] == "1":
            flips[row["arm"]] = flips.get(row["arm"], 0) + 1
    print(f"  went first: " + ", ".join(f"{a}x{n}" for a, n in sorted(flips.items()))
          + "  (balanced is best; the coin flip is random, not alternating)")

    # --- what the governor actually did ------------------------------------
    print("\nGOVERNOR TELEMETRY")
    seen = set()
    for row in rows:
        summary = gate_summary(os.path.join(outdir, row["log"]))
        key = (row["arm"], summary)
        if summary and key not in seen:
            seen.add(key)
            print(f"  {row['arm']:10} {summary}")
    if not seen:
        print("  (no governor active in either arm)")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(target):
        candidates = sorted(glob.glob("bench-results/ab-pilot-*"))
        target = candidates[-1] if candidates else target
    sys.exit(main(target))
