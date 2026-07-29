#!/usr/bin/env python3
"""Deterministic, disk-free tests for the opt-in N14 runtime path."""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_v2  # noqa: E402
import grouped_moe  # noqa: E402
import metal_moe  # noqa: E402


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  PASS {name}")


def test_config():
    print("configuration")
    check("default is off",
          grouped_moe.configured_group_size({}) == 0)
    check("off alias",
          grouped_moe.configured_group_size({"K3_MOE_GROUP_SIZE": "off"}) == 0)
    check("positive override",
          grouped_moe.configured_group_size({"K3_MOE_GROUP_SIZE": "4"}) == 4)
    try:
        grouped_moe.configured_group_size({"K3_MOE_GROUP_SIZE": "-2"})
    except ValueError:
        pass
    else:
        raise AssertionError("negative override rejected")
    print("  PASS invalid override rejected")


class FakeFetch:
    class GroupReadError(IOError):
        pass

    def __init__(self, available=True, fail=False):
        self.available = available
        self.fail = fail
        self.requested = []

    def begin_layer_groups(self, layer, eids, size):
        self.requested.append((layer, list(eids), size))
        if not self.available:
            return None

        def iterator():
            for start in range(0, len(eids), size):
                if self.fail and start:
                    raise self.GroupReadError("synthetic read failure")
                group = list(eids[start:start + size])
                yield group, {eid: f"raw-{eid}" for eid in group}

        return iterator()


class FakeMetal:
    def __init__(self):
        self.groups = []

    def moe_infer_grouped(self, x, groups, timing=None):
        for eids, weights, raw in groups:
            self.groups.append((list(eids), list(weights), dict(raw)))
        if timing is not None:
            timing.update(fetch_wait_s=0.25, kernel_s=0.5)
        return "grouped-output"


def test_orchestration():
    print("orchestration")
    old_size = grouped_moe.GROUP_SIZE
    try:
        grouped_moe.GROUP_SIZE = 2
        fetch = FakeFetch()
        metal = FakeMetal()
        x = torch.zeros((1, 8))
        record = {
            "ids": [[5, 2, 9, 1]],
            "weights": [[0.1, 0.4, 0.4, 0.2]],
        }
        result = grouped_moe.try_infer(x, record, 45, fetch, metal)
        check("grouped output returned", result[0] == "grouped-output")
        check("timing returned",
              result[1] == {"fetch_wait_s": 0.25, "kernel_s": 0.5})
        check("stable weight order",
              fetch.requested == [(45, [2, 9, 1, 5], 2)])
        check("all experts exactly once",
              [x for group in metal.groups for x in group[0]]
              == [2, 9, 1, 5])
        check("weights stay paired",
              [x for group in metal.groups for x in group[1]]
              == [0.4, 0.4, 0.2, 0.1])

        unavailable = FakeFetch(available=False)
        check("storage miss falls back",
              grouped_moe.try_infer(
                  x, record, 45, unavailable, FakeMetal()) is None)
        check("multi-token falls back before fetch",
              grouped_moe.try_infer(
                  torch.zeros((2, 8)),
                  {"ids": [[1, 2], [3, 4]],
                   "weights": [[0.6, 0.4], [0.7, 0.3]]},
                  45, fetch, FakeMetal()) is None)
        failed = FakeFetch(fail=True)
        check("read failure discards partial and falls back",
              grouped_moe.try_infer(x, record, 45, failed, FakeMetal()) is None)
        grouped_moe.GROUP_SIZE = 4
        check("full-size group preserves old path",
              grouped_moe.try_infer(x, record, 45, fetch, FakeMetal()) is None)
    finally:
        grouped_moe.GROUP_SIZE = old_size


class FakeFuture:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.result_calls = 0
        self.cancel_calls = 0

    def result(self):
        self.result_calls += 1
        if self.error:
            raise self.error
        return self.value

    def cancel(self):
        self.cancel_calls += 1
        return False


class FakeSlot:
    def __init__(self, eid):
        self.raw = {"eid": eid}


def fake_reader(futures):
    reader = object.__new__(fetch_v2._PreadReader)
    reader._lock = threading.Lock()
    reader._pending = {}
    retired = []
    reader._submit = lambda layer, eids: {
        eid: (FakeSlot(eid), futures[eid]) for eid in eids}
    reader._retire = lambda slots: retired.extend(slots)
    reader._drain = lambda jobs: None
    return reader, retired


def test_fetch_groups():
    print("fetch grouping")
    futures = {eid: FakeFuture(eid) for eid in (7, 3, 8, 2, 6)}
    reader, retired = fake_reader(futures)
    got = list(reader.read_layer_groups(45, [7, 3, 8, 2, 6], 2))
    check("yield boundaries preserved",
          [ids for ids, _raw in got] == [[7, 3], [8, 2], [6]])
    check("raw mapping preserved",
          [sorted(raw) for _ids, raw in got] == [[3, 7], [2, 8], [6]])
    check("every future awaited once",
          all(f.result_calls == 1 for f in futures.values()))
    check("all slots retired after consumption", len(retired) == 5)

    futures = {
        1: FakeFuture(1),
        2: FakeFuture(error=OSError("bad synthetic expert")),
        3: FakeFuture(3),
    }
    reader, retired = fake_reader(futures)
    try:
        list(reader.read_layer_groups(999, [1, 2, 3], 1))
    except fetch_v2.GroupReadError:
        pass
    else:
        raise AssertionError("read error must become GroupReadError")
    check("later future settled on failure", futures[3].result_calls == 1)
    check("all failure-path slots retired", len(retired) == 3)


def test_metal_accumulation():
    print("Metal accumulator contract (fake kernel)")
    old_load = metal_moe._load
    old_one = metal_moe._metal_one
    old_check = metal_moe.CHECK
    calls = []

    def fake_one(_lib, _x, eids, weights, _raw, output=None):
        calls.append(tuple(eids))
        value = np.float32(sum(
            np.float32(e + 1) * np.float32(w)
            for e, w in zip(eids, weights)))
        out = (np.full(metal_moe.HIDDEN, value, dtype=np.float32)
               if output is None else output)
        out[:] = value
        return out

    try:
        metal_moe._load = lambda: object()
        metal_moe._metal_one = fake_one
        metal_moe.CHECK = 0
        x = torch.zeros((1, metal_moe.HIDDEN), dtype=torch.float32)
        raw = {e: object() for e in (1, 3, 5, 7)}
        ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
        weights = torch.tensor([[0.5, 0.25, 0.125, 0.125]])
        mono = metal_moe.moe_infer_metal(x, ids, weights, raw)

        def groups():
            yield [1, 3], [0.5, 0.25], {1: raw[1], 3: raw[3]}
            yield [5, 7], [0.125, 0.125], {5: raw[5], 7: raw[7]}

        timing = {}
        grouped1 = metal_moe.moe_infer_grouped(x, groups(), timing)
        grouped2 = metal_moe.moe_infer_grouped(x, groups(), {})
        check("grouped equals monolithic", torch.equal(grouped1, mono))
        check("grouped repeat deterministic", torch.equal(grouped1, grouped2))
        check("one device result shape", tuple(grouped1.shape) == (1, 3584))
        check("timing fields populated",
              set(timing) == {"fetch_wait_s", "kernel_s"})
    finally:
        metal_moe._load = old_load
        metal_moe._metal_one = old_one
        metal_moe.CHECK = old_check


def main():
    test_config()
    test_orchestration()
    test_fetch_groups()
    test_metal_accumulation()
    print("\nALL GROUPED-MOE TESTS PASSED")


if __name__ == "__main__":
    main()
