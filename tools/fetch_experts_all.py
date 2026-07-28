#!/usr/bin/env python3
"""Bulk-download every routed expert, so no prompt is ever network-bound.

Deltafin streams experts on demand by default; this fetches the whole pool up
front (~1.45 TB, minus whatever is already cached). It is far more efficient
than on-demand fetching: each MoE layer's 896 experts are contiguous in one
shard, so this downloads them in large multi-expert runs instead of one request
per expert.

Resumable (skips anything already in k3-experts/), safe to interrupt, and it
refuses to start if the disk would end up under --min-free-gb.

    python tools/fetch_experts_all.py              # everything missing
    python tools/fetch_experts_all.py --layers 1-30
    python tools/fetch_experts_all.py --dry-run
"""
import argparse
import concurrent.futures
import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_v2 as fv  # noqa: E402

GROUP = int(os.environ.get("K3_BULK_GROUP", "8"))       # experts per request
WORKERS = int(os.environ.get("K3_BULK_WORKERS", "8"))   # parallel requests
NUM_EXPERTS = 896
_lock = threading.Lock()
_done_bytes = 0
_done_experts = 0


def missing_for_layer(layer):
    out = []
    for e in range(NUM_EXPERTS):
        p = os.path.join(fv.ECACHE, f"L{layer}-E{e}.bin")
        if os.path.exists(p) and os.path.getsize(p) == fv.EXPERT_SPAN:
            continue
        if os.path.exists(os.path.join(fv.ECACHE, f"L{layer}-E{e}.npz")):
            continue  # legacy cache entry, still usable
        out.append(e)
    return out


def runs(layer, eids):
    """Group missing experts into file-contiguous runs of at most GROUP."""
    spans = sorted((fv.expert_span(layer, e)[1], e) for e in eids)
    out, cur = [], []
    for start, e in spans:
        if cur and (len(cur) >= GROUP or start != cur[-1][0] + fv.EXPERT_SPAN):
            out.append(cur)
            cur = []
        cur.append((start, e))
    if cur:
        out.append(cur)
    return out


def fetch_run(layer, run):
    global _done_bytes, _done_experts
    shard, _, _ = fv.expert_span(layer, run[0][1])
    base = run[0][0]
    size = run[-1][0] + fv.EXPERT_SPAN - base
    for attempt in range(6):
        try:
            buf = fv._range_get(shard, base, size)
            break
        except Exception:
            if attempt == 5:
                print(f"  ! layer {layer} run at {base} failed permanently", flush=True)
                return
            time.sleep(2 * (attempt + 1))
    for start, e in run:
        off = start - base
        path = os.path.join(fv.ECACHE, f"L{layer}-E{e}.bin")
        tmp = f"{path}.tmp{os.getpid()}-{threading.get_ident()}"
        with open(tmp, "wb") as f:
            f.write(buf[off:off + fv.EXPERT_SPAN])
        os.replace(tmp, path)
    with _lock:
        _done_bytes += size
        _done_experts += len(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1-92", help="e.g. 1-92 or 5-10")
    ap.add_argument("--min-free-gb", type=float, default=100.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    lo, _, hi = args.layers.partition("-")
    layers = range(int(lo), int(hi or lo) + 1)

    plan = {L: missing_for_layer(L) for L in layers}
    n_missing = sum(len(v) for v in plan.values())
    need = n_missing * fv.EXPERT_SPAN
    free = shutil.disk_usage(fv.ECACHE).free
    print(f"missing {n_missing} experts across {len(layers)} layers = {need/1e12:.2f} TB")
    print(f"disk free {free/1e12:.2f} TB; after download {(free-need)/1e9:.0f} GB would remain")
    if n_missing == 0:
        print("nothing to do — the expert pool is fully cached.")
        return
    if free - need < args.min_free_gb * 1e9:
        print(f"REFUSING: would leave less than --min-free-gb {args.min_free_gb:.0f} GB free.")
        sys.exit(1)
    if args.dry_run:
        print("(dry run; nothing fetched)")
        return

    jobs = [(L, r) for L in layers for r in runs(L, plan[L])]
    print(f"{len(jobs)} requests, {GROUP} experts each, {WORKERS} in parallel", flush=True)
    t0 = time.time()
    last = [t0]
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_run, L, r) for L, r in jobs]
        for _ in concurrent.futures.as_completed(futs):
            now = time.time()
            if now - last[0] > 30:
                last[0] = now
                mbs = _done_bytes / 1e6 / max(now - t0, 1)
                left = (need - _done_bytes) / 1e6 / max(mbs, 1)
                print(f"  {_done_experts}/{n_missing} experts  {_done_bytes/1e12:.2f} TB  "
                      f"{mbs:.0f} MB/s  eta {left/3600:.1f} h", flush=True)
    print(f"done: {_done_experts} experts in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
