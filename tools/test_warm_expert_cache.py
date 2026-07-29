#!/usr/bin/env python3
"""Q8 exact conversion and frequency-planner tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import warm_expert_cache as warmer


def main() -> int:
    real = next(warmer.DEFAULT_CACHE.glob("L1-E*.bin"))
    original = real.read_bytes()
    assert len(original) == warmer.EXPERT_SPAN
    with tempfile.TemporaryDirectory(prefix="k3-warm-test-") as tmp:
        cache = Path(tmp)
        arrays = {}
        offset = 0
        for name, nbytes, shape in warmer.LAYOUT:
            arrays[name] = np.frombuffer(
                original, dtype=np.uint8, count=nbytes,
                offset=offset).reshape(shape)
            offset += nbytes
        np.savez(cache / "L1-E7.npz", **arrays)
        destination = warmer.convert_one(cache / "L1-E7.npz", cache)
        assert destination.read_bytes() == original
        print("PASS NPZ-to-raw conversion is byte-exact")

        trace = cache / "trace.jsonl"
        trace.write_text(
            json.dumps({"layer": 1, "ids": [7, 8, 8, 9]}) + "\n"
            + json.dumps({"layer": 1, "ids": [8, 9]}) + "\n")
        counts = warmer.trace_counts([trace])
        rows = warmer.plan(counts, cache)
        assert rows == [(3, 1, 8), (2, 1, 9)]
        print("PASS frequency ordering and present-entry exclusion")
        assert warmer.plan(counts, cache, limit=1) == [(3, 1, 8)]
        print("PASS bounded plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
