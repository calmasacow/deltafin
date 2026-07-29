#!/usr/bin/env python3
"""Focused, weight-free gates for the portable X12 routing record.

The native CPU modules are loaded against fake libraries, and the production
``moe_infer_lazy`` function is extracted from its AST so this test never loads
the model, expert files, Metal, CUDA, or ROCm.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import time
import types
import unittest
from unittest import mock

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import routing_record as routing_records  # noqa: E402
import runtime_platform  # noqa: E402


class FakeFunction:
    def __init__(self, result=None):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *_args):
        return self.result


class FakeLibrary:
    def __init__(self, symbols):
        self.mxfp4_abi_version = FakeFunction(
            runtime_platform.NATIVE_ABI_VERSION)
        for symbol in symbols:
            setattr(self, symbol, FakeFunction())


def load_native_module(filename):
    def fake_loader(_directory, stem, **kwargs):
        return FakeLibrary(kwargs["required_symbols"]), f"/fake/{stem}"

    name = f"_x12_{filename.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(
            runtime_platform, "load_native_library", fake_loader):
        spec.loader.exec_module(module)
    return module


class ConversionState:
    def __init__(self):
        self.to_calls = []
        self.tolist_calls = 0


class RouteTensorSpy:
    """Tensor facade counting only route materialization operations."""

    def __init__(self, tensor, state=None):
        self.tensor = tensor
        self.state = state if state is not None else ConversionState()

    def to(self, *args, **kwargs):
        self.state.to_calls.append((args, kwargs))
        return RouteTensorSpy(
            self.tensor.to(*args, **kwargs), self.state)

    def tolist(self):
        self.state.tolist_calls += 1
        return self.tensor.tolist()

    def view(self, *shape):
        return RouteTensorSpy(self.tensor.view(*shape), self.state)


def route_fixture(tokens):
    if tokens == 1:
        ids = torch.tensor([[7, 3, 9]], dtype=torch.int32)
        # Equal first two weights explicitly exercise stable route-slot ties.
        weights = torch.tensor(
            [[0.25, 0.25, 0.5]], dtype=torch.float64)
    else:
        ids = torch.tensor(
            [[7, 3, 9], [4, 8, 2]], dtype=torch.int64)
        weights = torch.tensor(
            [[0.25, 0.25, 0.5], [0.125, 0.625, 0.25]],
            dtype=torch.float16,
        )
    return ids, weights


def install_fake_experts(module, *, batch):
    calls = []
    if batch:
        def fake_set(raws, x, nthreads=None):
            calls.append((list(raws), nthreads))
            return np.stack([
                np.full_like(x, np.float32(raw), dtype=np.float32)
                for raw in raws
            ])
        module.expert_set_ffn = fake_set
    else:
        def fake_one(raw, x):
            calls.append(raw)
            return np.full_like(x, np.float32(raw), dtype=np.float32)
        module.expert_ffn = fake_one
    return calls


def expected_output(ids, weights, width):
    rows = []
    for id_row, weight_row in zip(ids, weights):
        value = np.float32(0.0)
        for expert, weight in zip(id_row, weight_row):
            value += np.float32(weight) * np.float32(expert)
        rows.append(np.full(width, value, dtype=np.float32))
    return np.stack(rows)


def extract_moe_infer_lazy(namespace):
    tree = ast.parse((HERE / "kimi_run.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "moe_infer_lazy"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(HERE / "kimi_run.py"), "exec"), namespace)
    return namespace["moe_infer_lazy"]


class TraceSpy:
    def __init__(self, enabled):
        self.enabled = enabled
        self.calls = []

    def record(self, step, layer, ids, weights):
        if not self.enabled:
            return
        # Mirror RouterTrace's relevant distinction: lists are consumed
        # directly; tensors would require another materialization.
        rows = (
            weights
            if isinstance(weights, list)
            else weights.view(-1).tolist()
        )
        self.calls.append((step, layer, list(ids), rows))


class FetchSpy:
    def __init__(self):
        self.calls = []

    def fetch_experts(self, layer, ids, dequant):
        self.calls.append((layer, list(ids), dequant))
        return {}


def driver_namespace(*, fast, trace_enabled, backend):
    fetch = FetchSpy()
    trace = TraceSpy(trace_enabled)
    pilot = types.SimpleNamespace(
        enabled=lambda: False,
        on_actual=lambda *_args: None,
    )
    namespace = {
        "_step_ctx": {"layer": 17, "step": 23},
        "routing_records": routing_records,
        "FAST_MOE": fast,
        "EXPERT_SEL": {"layer_calls": 0, "uniq": 0, "pos": 0},
        "_LAST_SEL": {},
        "pilot": pilot,
        "GROUPED_MOE_ACTIVE": False,
        "grouped_moe": types.SimpleNamespace(try_infer=lambda *_args: None),
        "fetch_v2": object(),
        "metal_moe": object(),
        "time": time,
        "k3loader": fetch,
        "_issue_next_expert_prefetch": lambda _layer: None,
        "TIMES": {"expert_fetch": 0.0, "moe_kernel": 0.0},
        "TRACE": trace,
        "_MOE_FN": backend,
        "set_param": lambda *_args: None,
        "torch": torch,
        "_orig_moe_infer": lambda *_args: "slow-result",
    }
    return namespace, fetch, trace


class RoutingRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fast = load_native_module("fast_moe.py")
        cls.batch = load_native_module("fast_moe_batch.py")

    def check_backend(self, module, tokens, *, batch):
        ids_tensor, weights_tensor = route_fixture(tokens)
        expected_ids = ids_tensor.tolist()
        expected_weights = weights_tensor.to(torch.float32).tolist()
        raw = {
            expert: expert
            for row in expected_ids
            for expert in row
        }
        x = torch.ones((tokens, 4), dtype=torch.float32)

        ids = RouteTensorSpy(ids_tensor)
        weights = RouteTensorSpy(weights_tensor)
        rows = ids.tolist()
        record = routing_records.materialize(
            ids, weights, ids=rows)
        self.assertIs(record["ids"], rows)
        self.assertEqual(record["ids"], expected_ids)
        self.assertEqual(record["weights"], expected_weights)
        self.assertEqual(ids.state.tolist_calls, 1)
        self.assertEqual(weights.state.tolist_calls, 1)
        self.assertEqual(len(weights.state.to_calls), 1)
        self.assertEqual(weights.state.to_calls[0][0], (torch.float32,))

        calls = install_fake_experts(module, batch=batch)
        if batch:
            got = module.moe_infer_fast(
                x, ids, weights, raw, nthreads=3,
                routing_record=record)
            self.assertEqual(
                calls,
                [(row, 3) for row in expected_ids],
            )
        else:
            got = module.moe_infer_fast(
                x, ids, weights, raw, routing_record=record)
            self.assertEqual(
                calls,
                [expert for row in expected_ids for expert in row],
            )
        # The backend consumed the exact lists without touching either route
        # tensor again.
        self.assertEqual(ids.state.tolist_calls, 1)
        self.assertEqual(weights.state.tolist_calls, 1)
        self.assertEqual(len(weights.state.to_calls), 1)
        np.testing.assert_array_equal(
            got.numpy(),
            expected_output(expected_ids, expected_weights, x.shape[1]),
        )

        # Omit the record to prove the longstanding public call remains valid.
        legacy_ids = RouteTensorSpy(ids_tensor)
        legacy_weights = RouteTensorSpy(weights_tensor)
        calls.clear()
        if batch:
            legacy = module.moe_infer_fast(
                x, legacy_ids, legacy_weights, raw, 5)
            self.assertEqual(calls, [(row, 5) for row in expected_ids])
        else:
            legacy = module.moe_infer_fast(
                x, legacy_ids, legacy_weights, raw)
            self.assertEqual(
                calls,
                [expert for row in expected_ids for expert in row],
            )
        self.assertEqual(legacy_ids.state.tolist_calls, 1)
        self.assertEqual(legacy_weights.state.tolist_calls, 1)
        self.assertEqual(len(legacy_weights.state.to_calls), 1)
        self.assertTrue(torch.equal(got, legacy))

    def test_fast_moe_record_n1_and_n2(self):
        for tokens in (1, 2):
            with self.subTest(tokens=tokens):
                self.check_backend(self.fast, tokens, batch=False)

    def test_fast_moe_batch_record_n1_and_n2(self):
        for tokens in (1, 2):
            with self.subTest(tokens=tokens):
                self.check_backend(self.batch, tokens, batch=True)

    def test_driver_one_conversion_trace_off_and_on(self):
        for tokens in (1, 2):
            for trace_enabled in (False, True):
                with self.subTest(tokens=tokens, trace=trace_enabled):
                    ids_tensor, weights_tensor = route_fixture(tokens)
                    ids = RouteTensorSpy(ids_tensor)
                    weights = RouteTensorSpy(weights_tensor)
                    captured = []

                    def backend(
                            x, topk_ids, topk_weight, raw,
                            routing_record=None):
                        captured.append(
                            (x, topk_ids, topk_weight, raw, routing_record))
                        return "fast-result"

                    namespace, fetch, trace = driver_namespace(
                        fast=True,
                        trace_enabled=trace_enabled,
                        backend=backend,
                    )
                    infer = extract_moe_infer_lazy(namespace)
                    marker = object()
                    self.assertEqual(
                        infer(None, marker, ids, weights), "fast-result")
                    self.assertEqual(ids.state.tolist_calls, 1)
                    self.assertEqual(weights.state.tolist_calls, 1)
                    self.assertEqual(len(weights.state.to_calls), 1)
                    record = captured[0][4]
                    self.assertEqual(record["ids"], ids_tensor.tolist())
                    self.assertEqual(
                        record["weights"],
                        weights_tensor.to(torch.float32).tolist(),
                    )
                    self.assertEqual(
                        fetch.calls,
                        [(
                            17,
                            sorted(set(ids_tensor.view(-1).tolist())),
                            False,
                        )],
                    )
                    if trace_enabled:
                        self.assertEqual(len(trace.calls), 1)
                        self.assertIs(trace.calls[0][3], record["weights"])
                    else:
                        self.assertEqual(trace.calls, [])

    def test_driver_backend_failure_propagates_without_reconversion(self):
        ids_tensor, weights_tensor = route_fixture(2)
        ids = RouteTensorSpy(ids_tensor)
        weights = RouteTensorSpy(weights_tensor)
        records = []

        class BackendFailure(RuntimeError):
            pass

        def backend(
                _x, _ids, _weights, _raw, routing_record=None):
            records.append(routing_record)
            raise BackendFailure("expected backend failure")

        namespace, fetch, trace = driver_namespace(
            fast=True, trace_enabled=True, backend=backend)
        infer = extract_moe_infer_lazy(namespace)
        with self.assertRaisesRegex(BackendFailure, "expected backend"):
            infer(None, object(), ids, weights)
        self.assertEqual(ids.state.tolist_calls, 1)
        self.assertEqual(weights.state.tolist_calls, 1)
        self.assertEqual(len(weights.state.to_calls), 1)
        self.assertEqual(len(records), 1)
        self.assertIs(trace.calls[0][3], records[0]["weights"])
        self.assertEqual(len(fetch.calls), 1)

    def test_slow_fallback_retains_trace_off_and_on_behavior(self):
        for trace_enabled in (False, True):
            with self.subTest(trace=trace_enabled):
                ids_tensor, weights_tensor = route_fixture(1)
                ids = RouteTensorSpy(ids_tensor)
                weights = RouteTensorSpy(weights_tensor)

                def forbidden_backend(*_args, **_kwargs):
                    raise AssertionError(
                        "fast backend called from slow fallback")

                namespace, fetch, trace = driver_namespace(
                    fast=False,
                    trace_enabled=trace_enabled,
                    backend=forbidden_backend,
                )
                infer = extract_moe_infer_lazy(namespace)
                owner = types.SimpleNamespace(
                    experts={
                        expert: object()
                        for expert in ids_tensor.view(-1).tolist()
                    }
                )
                self.assertEqual(
                    infer(owner, object(), ids, weights), "slow-result")
                self.assertEqual(ids.state.tolist_calls, 1)
                # TRACE=off retains zero weight materialization. TRACE=on
                # retains the old trace-owned one tolist without an fp32 cast.
                self.assertEqual(
                    weights.state.tolist_calls,
                    1 if trace_enabled else 0,
                )
                self.assertEqual(weights.state.to_calls, [])
                self.assertEqual(
                    len(trace.calls), 1 if trace_enabled else 0)
                self.assertEqual(fetch.calls[0][2], True)

    def test_driver_wiring_has_no_backend_or_platform_branch(self):
        source = (HERE / "kimi_run.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "moe_infer_lazy"
        )
        text = ast.unparse(function)
        self.assertEqual(text.count("topk_ids.tolist()"), 1)
        self.assertIn("routing_records.materialize", text)
        self.assertIn("if FAST_MOE else None", text)
        self.assertIn("routing_record=routing_record", text)
        self.assertNotIn("MOE_BACKEND", text)
        self.assertNotIn("sys.platform", text)
        self.assertNotIn("platform.", text)


if __name__ == "__main__":
    unittest.main()
