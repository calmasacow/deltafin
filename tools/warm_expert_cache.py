#!/usr/bin/env python3
"""Q8 idle-time expert-cache planner and legacy-NPZ converter.

The default is read-only: rank absent raw experts by observed route frequency.
``--convert-npz`` atomically converts legacy cache entries to the zero-parse raw
span used by the fast reader. ``--fetch N`` downloads at most N planned experts
through the established fetcher; it is explicit because it uses network and
disk. Run this between generations, never on the token hot path.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "k3-experts"
DEFAULT_TRACES = sorted((ROOT / "k3-meta").glob("router_trace*.jsonl"))
EXPERT_SPAN = 17_547_264
LAYOUT = (
    ("w1_p", 5_505_024, (3072, 1792)),
    ("w1_s", 344_064, (3072, 112)),
    ("w2_p", 5_505_024, (3584, 1536)),
    ("w2_s", 344_064, (3584, 96)),
    ("w3_p", 5_505_024, (3072, 1792)),
    ("w3_s", 344_064, (3072, 112)),
)
TAG = re.compile(r"L([0-9]+)-E([0-9]+)\.npz$")


def trace_counts(paths: list[Path]) -> collections.Counter[tuple[int, int]]:
    counts: collections.Counter[tuple[int, int]] = collections.Counter()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    layer = int(row["layer"])
                    ids = row["ids"]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: invalid trace row") from exc
                for eid in ids:
                    counts[(layer, int(eid))] += 1
    return counts


def raw_path(cache: Path, layer: int, eid: int) -> Path:
    return cache / f"L{layer}-E{eid}.bin"


def valid_raw(cache: Path, layer: int, eid: int) -> bool:
    try:
        return raw_path(cache, layer, eid).stat().st_size == EXPERT_SPAN
    except OSError:
        return False


def plan(
        counts: collections.Counter[tuple[int, int]], cache: Path,
        limit: int | None = None,
) -> list[tuple[int, int, int]]:
    rows = [
        (frequency, layer, eid)
        for (layer, eid), frequency in counts.items()
        if not valid_raw(cache, layer, eid)
    ]
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    if limit is not None:
        rows = rows[:max(0, limit)]
    return rows


def convert_one(path: Path, cache: Path) -> Path:
    match = TAG.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not an expert NPZ name: {path}")
    layer, eid = map(int, match.groups())
    destination = raw_path(cache, layer, eid)
    if valid_raw(cache, layer, eid):
        return destination
    temporary = destination.with_name(
        destination.name + f".tmp{os.getpid()}")
    try:
        total = 0
        with np.load(path, allow_pickle=False) as archive, \
                temporary.open("xb") as output:
            for name, nbytes, shape in LAYOUT:
                array = archive[name]
                if (array.dtype != np.uint8 or tuple(array.shape) != shape
                        or array.nbytes != nbytes):
                    raise ValueError(
                        f"{path}:{name} has {array.dtype}/{array.shape}/"
                        f"{array.nbytes}, expected uint8/{shape}/{nbytes}")
                contiguous = np.ascontiguousarray(array)
                output.write(memoryview(contiguous).cast("B"))
                total += contiguous.nbytes
            output.flush()
            os.fsync(output.fileno())
        if total != EXPERT_SPAN or temporary.stat().st_size != EXPERT_SPAN:
            raise ValueError(f"{path}: converted size {total} != {EXPERT_SPAN}")
        os.replace(temporary, destination)
        return destination
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def convert_all(cache: Path) -> tuple[int, int]:
    converted = 0
    already = 0
    for path in sorted(cache.glob("L*-E*.npz")):
        match = TAG.fullmatch(path.name)
        if match and valid_raw(cache, *map(int, match.groups())):
            already += 1
            continue
        convert_one(path, cache)
        converted += 1
    return converted, already


def fetch(rows: list[tuple[int, int, int]]) -> None:
    # Import only for the explicitly mutating/networked mode. DELTAFIN_ROOT
    # continues to choose the same cache as ordinary inference.
    sys.path.insert(0, str(ROOT / "tools"))
    import fetch_v2
    for index, (frequency, layer, eid) in enumerate(rows, 1):
        print(
            f"[{index}/{len(rows)}] L{layer}-E{eid} "
            f"(trace frequency {frequency})", flush=True)
        fetch_v2.fetch_expert_raw(layer, eid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--traces", type=Path, nargs="*", default=DEFAULT_TRACES)
    parser.add_argument("--convert-npz", action="store_true")
    parser.add_argument("--fetch", type=int, default=0,
                        help="explicitly download at most N planned experts")
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.fetch < 0 or args.show < 0:
        parser.error("--fetch and --show must be non-negative")

    converted = already = 0
    if args.convert_npz:
        converted, already = convert_all(args.cache)
    counts = trace_counts(list(args.traces))
    rows = plan(counts, args.cache)
    payload = {
        "trace_files": [str(path) for path in args.traces],
        "observed_routes": sum(counts.values()),
        "unique_observed_experts": len(counts),
        "missing_observed_experts": len(rows),
        "missing_observed_bytes": len(rows) * EXPERT_SPAN,
        "converted_npz": converted,
        "already_raw": already,
        "top_missing": [
            {"frequency": frequency, "layer": layer, "expert": eid}
            for frequency, layer, eid in rows[:args.show]
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"routes={payload['observed_routes']} "
            f"unique={payload['unique_observed_experts']} "
            f"missing={payload['missing_observed_experts']} "
            f"({payload['missing_observed_bytes']/1e9:.3f} GB) "
            f"converted={converted}", flush=True)
        for row in payload["top_missing"]:
            print(
                f"  L{row['layer']}-E{row['expert']}: "
                f"{row['frequency']} observations")
    if args.fetch:
        fetch(rows[:args.fetch])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
