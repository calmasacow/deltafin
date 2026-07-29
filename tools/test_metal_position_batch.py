#!/usr/bin/env python3
"""Disk-free contract tests for the opt-in N5 Metal position batch.

No Metal device, dylib, model weights, or expert cache is opened. The fake C ABI
checks the flattened routing layout and writes a deterministic output through
the same ctypes pointers used by the real bridge.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metal_moe  # noqa: E402


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  PASS {name}")


class FakeBatchLibrary:
    def __init__(self, expected, events):
        self.expected = expected
        self.events = events
        self.calls = 0

    def k3_metal_moe_positions(
            self, ptrs, n_edges, offsets, n_positions, weights, x, out):
        self.calls += 1
        self.events.append("abi")
        got_ptrs = [ptrs[i] for i in range(n_edges)]
        got_offsets = [offsets[i] for i in range(n_positions + 1)]
        got_weights = np.ctypeslib.as_array(weights, shape=(n_edges,)).copy()
        got_x = np.ctypeslib.as_array(
            x, shape=(n_positions * metal_moe.HIDDEN,)).copy()
        assert got_ptrs == self.expected["ptrs"]
        assert got_offsets == self.expected["offsets"]
        np.testing.assert_array_equal(
            got_weights, np.asarray(self.expected["weights"], dtype=np.float32))
        np.testing.assert_array_equal(got_x, self.expected["x"].reshape(-1))

        out_view = np.ctypeslib.as_array(
            out, shape=(n_positions * metal_moe.HIDDEN,))
        out_view[:] = got_x
        for position in range(n_positions):
            begin = position * metal_moe.HIDDEN
            end = begin + metal_moe.HIDDEN
            out_view[begin:end] += got_offsets[position + 1] - got_offsets[position]
        return 0


def test_position_batch_contract():
    print("position-batch ctypes and lifetime contract")
    ids = [[3, 7], [7, 9, 3]]
    weights = [[0.25, 0.75], [0.1, 0.2, 0.7]]
    x = (np.arange(2 * metal_moe.HIDDEN, dtype=np.float32)
         .reshape(2, metal_moe.HIDDEN) / np.float32(1024.0))
    expected = {
        "ptrs": [1003, 1007, 1007, 1009, 1003],
        "offsets": [0, 2, 5],
        "weights": [0.25, 0.75, 0.1, 0.2, 0.7],
        "x": x,
    }
    events = []
    spans = []
    syncs = []
    trims = []
    owners = {eid: object() for eid in (3, 7, 9)}
    lib = FakeBatchLibrary(expected, events)

    def fake_span(raw, slot):
        spans.append((raw, slot))
        events.append(f"span-{raw}")
        return 1000 + raw, owners[raw]

    def fake_sync(ptr, owner):
        syncs.append((ptr, owner))
        events.append(f"sync-{ptr}")

    def fake_trim(protect):
        trims.append(set(protect))
        events.append("trim")

    old_span = metal_moe._span_ptr
    old_sync = metal_moe._sync_wrap
    old_trim = metal_moe._trim
    try:
        metal_moe._span_ptr = fake_span
        metal_moe._sync_wrap = fake_sync
        metal_moe._trim = fake_trim
        got = metal_moe._metal_positions(
            lib, x, ids, weights, {eid: eid for eid in owners})
    finally:
        metal_moe._span_ptr = old_span
        metal_moe._sync_wrap = old_sync
        metal_moe._trim = old_trim

    expected_out = x + np.asarray([[2], [3]], dtype=np.float32)
    np.testing.assert_array_equal(got, expected_out)
    check("one C ABI call", lib.calls == 1)
    check("unique expert staging slots",
          spans == [(3, 0), (7, 1), (9, 2)])
    check("each unique owner pinned once",
          syncs == [(1003, owners[3]), (1007, owners[7]),
                    (1009, owners[9])])
    check("all call pointers protected",
          trims == [{1003, 1007, 1009}])
    check("trim happens only after synchronous ABI returns",
          events.index("trim") > events.index("abi"))


class LegacyLibrary:
    pass


def test_opt_in_and_legacy_fallback():
    print("opt-in selection and legacy fallback")
    ids = torch.tensor([[3, 7], [9, 3]], dtype=torch.long)
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    x = torch.zeros((2, metal_moe.HIDDEN), dtype=torch.float32)
    raw = {eid: object() for eid in (3, 7, 9)}
    calls = []

    def fake_one(_lib, _x, eids, wts, _raw, output=None):
        calls.append((list(eids), list(wts)))
        output[:] = np.float32(len(calls))
        return output

    old_load = metal_moe._load
    old_one = metal_moe._metal_one
    old_batch = metal_moe.POSITION_BATCH
    try:
        metal_moe._metal_one = fake_one

        # The new symbol alone never changes the default.
        metal_moe._load = lambda: FakeBatchLibrary({}, [])
        metal_moe.POSITION_BATCH = False
        got = metal_moe.moe_infer_metal(x, ids, weights, raw)
        check("default remains sequential", len(calls) == 2)
        check("sequential row order retained",
              calls[0][0] == [3, 7] and calls[1][0] == [9, 3])
        check("sequential output retained",
              torch.equal(got[:, 0], torch.tensor([1.0, 2.0])))

        # Opting in against an old installed dylib is also a safe fallback.
        calls.clear()
        metal_moe._load = lambda: LegacyLibrary()
        metal_moe.POSITION_BATCH = True
        got = metal_moe.moe_infer_metal(x, ids, weights, raw)
        check("missing batch ABI falls back", len(calls) == 2)
        check("legacy fallback output retained",
              torch.equal(got[:, 0], torch.tensor([1.0, 2.0])))
    finally:
        metal_moe._load = old_load
        metal_moe._metal_one = old_one
        metal_moe.POSITION_BATCH = old_batch


def _live_routes(path, layer=1):
    first = None
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("layer") != layer:
                continue
            if row.get("step") == 1:
                first = row
            elif row.get("step") == 2 and first is not None:
                return [first, row]
    raise RuntimeError(f"{path} has no layer-{layer} step-1/2 route pair")


def test_live(rounds):
    """Smallest real N5 gate: exact T=2 parity, then warmed API timing."""
    print("\nlive Metal T=2 gate")
    if not metal_moe.available():
        raise RuntimeError(f"Metal unavailable: {metal_moe.last_error()}")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    routes = _live_routes(os.path.join(root, "k3-meta", "router_trace.jsonl"))
    ids_list = [row["ids"] for row in routes]
    weights_list = [row["w"] for row in routes]
    all_ids = sorted({eid for row in ids_list for eid in row})
    raw = {}
    for eid in all_ids:
        expert = metal_moe.raw_from_cache(
            1, eid, root=os.path.join(root, "k3-experts"))
        if expert is None:
            raise RuntimeError(f"missing full L1-E{eid}.bin")
        raw[eid] = expert

    ids = torch.tensor(ids_list, dtype=torch.long)
    weights = torch.tensor(weights_list, dtype=torch.float32)
    rng = np.random.default_rng(20260728)
    x = torch.from_numpy(
        (rng.standard_normal((2, metal_moe.HIDDEN)) * 0.25).astype(np.float32))
    record = {"ids": ids_list, "weights": weights_list}

    def run(batch):
        old = metal_moe.POSITION_BATCH
        try:
            metal_moe.POSITION_BATCH = batch
            return metal_moe.moe_infer_metal(
                x, ids, weights, raw, routing_record=record)
        finally:
            metal_moe.POSITION_BATCH = old

    original_mode = metal_moe.stats()["bindless"]
    tested_modes = set()
    try:
        for requested in (0, 1):
            actual = metal_moe.set_mode(requested)
            if actual in tested_modes:
                continue
            tested_modes.add(actual)
            sequential = run(False)
            batched = run(True)
            max_abs = float((sequential - batched).abs().max())
            check(f"mode {actual} exact T=2 parity",
                  torch.equal(sequential, batched))
            print(f"    max_abs={max_abs:.3e}")
    finally:
        metal_moe.set_mode(original_mode)

    # Warm both arms after restoring the runtime-selected mode. Alternate order
    # to keep launch-order and drift bias out of the small integration sample.
    run(False)
    run(True)
    samples = {False: [], True: []}
    for iteration in range(rounds):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        for batch in order:
            start = time.perf_counter()
            run(batch)
            samples[batch].append((time.perf_counter() - start) * 1000.0)
    seq_ms = statistics.median(samples[False])
    batch_ms = statistics.median(samples[True])
    speedup = seq_ms / batch_ms
    print(f"  mode={metal_moe.stats()['bindless']} experts={len(all_ids)} "
          f"rounds={rounds}")
    print(f"  sequential median {seq_ms:.3f} ms")
    print(f"  position-batch median {batch_ms:.3f} ms")
    print(f"  speedup {speedup:.4f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true",
        help="also run a real Metal T=2 parity and timing gate")
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    test_position_batch_contract()
    test_opt_in_and_legacy_fallback()
    print("\nALL METAL POSITION-BATCH TESTS PASSED")
    if args.live:
        test_live(args.rounds)


if __name__ == "__main__":
    main()
