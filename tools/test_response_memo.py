#!/usr/bin/env python3
"""Deterministic tests and a tiny Q14 replay-cost falsifier."""
from __future__ import annotations

import time

from response_memo import DeterministicResponseMemo


def require(condition, label):
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def main():
    memo = DeterministicResponseMemo(2)
    prompt = [1008, 10484, 318, 15383, 387]
    response = [17374, 13, 646]
    memo.put("completion", prompt, 3, response, " Paris. The", "length")
    hit = memo.get("completion", prompt, 3)
    require(hit is not None, "exact request hits")
    require(list(hit.token_ids) == response, "tokens replay exactly")
    require(hit.text == " Paris. The", "text replays exactly")
    require(memo.get("chat", prompt, 3) is None, "mode is in key")
    require(memo.get("completion", prompt, 2) is None, "limit is in key")
    require(memo.get("completion", prompt + [13], 3) is None,
            "nearby prefix cannot alias")

    memo.put("completion", [1], 1, [2], "a", "length")
    memo.put("completion", [3], 1, [4], "b", "length")
    require(len(memo) == 2, "entry limit enforced")
    require(memo.get("completion", prompt, 3) is None, "LRU entry evicted")

    off = DeterministicResponseMemo(0)
    off.put("completion", prompt, 3, response, "x", "length")
    require(off.get("completion", prompt, 3) is None, "zero disables cache")

    memo = DeterministicResponseMemo(4)
    memo.put("completion", prompt, 3, response, " Paris. The", "length")
    reps = 100_000
    before = time.perf_counter_ns()
    for _ in range(reps):
        if memo.get("completion", prompt, 3) is None:
            raise AssertionError("unexpected miss")
    ns = time.perf_counter_ns() - before
    print(f"Q14 exact replay lookup: {ns / reps:.1f} ns/hit ({reps} hits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
