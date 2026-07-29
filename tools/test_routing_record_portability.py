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


def extract_driver_function(name, namespace):
    tree = ast.parse((HERE / "kimi_run.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(HERE / "kimi_run.py"), "exec"), namespace)
    return namespace[name]


def extract_moe_infer_lazy(namespace):
    return extract_driver_function("moe_infer_lazy", namespace)


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
        return {expert: expert for expert in ids}


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
        "_CPU_MOE_FN": backend,
        "_CUDA_MOE_ACTIVE": False,
        "_CUDA_MODEL_KEY": "/synthetic/model:4:2",
        "cuda_moe": None,
        "DEV": torch.device("cpu"),
        "PROFILE": False,
        "_device_synchronize": lambda: None,
        "_disable_cuda_moe": lambda _reason: None,
        "_slow_moe_infer": lambda *_args: "slow-result",
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

    def test_cuda_cache_hits_skip_fetch_and_keep_layer_identity(self):
        ids_tensor, weights_tensor = route_fixture(1)
        ids = RouteTensorSpy(ids_tensor)
        weights = RouteTensorSpy(weights_tensor)
        backend_calls = []
        cache_calls = []

        def backend(
                x, topk_ids, topk_weight, raw, routing_record=None,
                *, layer_index, model_key):
            backend_calls.append((
                x, topk_ids, topk_weight, raw, routing_record,
                layer_index, model_key,
            ))
            return "cuda-result"

        namespace, fetch, _trace = driver_namespace(
            fast=True, trace_enabled=False, backend=backend)
        namespace["_CUDA_MOE_ACTIVE"] = True
        namespace["DEV"] = torch.device("cuda:1")
        namespace["cuda_moe"] = types.SimpleNamespace(
            missing_experts=lambda layer, experts, device, model_key=None: (
                cache_calls.append((
                    layer, list(experts), device, model_key
                ))
                or [3, 9]
            )
        )
        infer = extract_moe_infer_lazy(namespace)
        self.assertEqual(
            infer(None, object(), ids, weights), "cuda-result")
        self.assertEqual(
            cache_calls,
            [(17, [3, 7, 9], torch.device("cuda:1"),
              "/synthetic/model:4:2")],
        )
        self.assertEqual(fetch.calls, [(17, [3, 9], False)])
        self.assertEqual(set(backend_calls[0][3]), {3, 9})
        self.assertEqual(backend_calls[0][5:], (
            17, "/synthetic/model:4:2"))

    def test_cuda_failure_refetches_complete_route_for_cpu(self):
        ids_tensor, weights_tensor = route_fixture(1)
        ids = RouteTensorSpy(ids_tensor)
        weights = RouteTensorSpy(weights_tensor)
        cpu_calls = []

        def failing_cuda(*_args, **_kwargs):
            raise RuntimeError("synthetic launch failure")

        namespace, fetch, _trace = driver_namespace(
            fast=True, trace_enabled=False, backend=failing_cuda)
        namespace["_CUDA_MOE_ACTIVE"] = True
        namespace["DEV"] = torch.device("cuda:0")
        namespace["cuda_moe"] = types.SimpleNamespace(
            missing_experts=lambda *_args, **_kwargs: []
        )

        def disable(_reason):
            namespace["_CUDA_MOE_ACTIVE"] = False

        def cpu_backend(
                _x, _ids, _weights, raw, routing_record=None):
            cpu_calls.append((raw, routing_record))
            return "cpu-result"

        namespace["_disable_cuda_moe"] = disable
        namespace["_CPU_MOE_FN"] = cpu_backend
        infer = extract_moe_infer_lazy(namespace)
        self.assertEqual(infer(None, object(), ids, weights), "cpu-result")
        # The all-hit CUDA query does no initial read. Once CUDA fails, the
        # complete unique route—not a partial miss set—is fetched for CPU.
        self.assertEqual(fetch.calls, [(17, [3, 7, 9], False)])
        self.assertEqual(len(cpu_calls), 1)
        self.assertEqual(ids.state.tolist_calls, 1)
        self.assertEqual(weights.state.tolist_calls, 1)

    def test_cuda_cache_query_failure_uses_complete_cpu_route(self):
        ids_tensor, weights_tensor = route_fixture(1)
        ids = RouteTensorSpy(ids_tensor)
        weights = RouteTensorSpy(weights_tensor)
        cpu_calls = []

        def cpu_backend(
                _x, _ids, _weights, raw, routing_record=None):
            cpu_calls.append((raw, routing_record))
            return "cpu-after-query-failure"

        namespace, fetch, _trace = driver_namespace(
            fast=True, trace_enabled=False, backend=cpu_backend)
        namespace["_CUDA_MOE_ACTIVE"] = True
        namespace["DEV"] = torch.device("cuda:0")

        def query_failure(*_args, **_kwargs):
            raise RuntimeError("synthetic cache corruption")

        namespace["cuda_moe"] = types.SimpleNamespace(
            missing_experts=query_failure)

        def disable(_reason):
            namespace["_CUDA_MOE_ACTIVE"] = False

        namespace["_disable_cuda_moe"] = disable
        infer = extract_moe_infer_lazy(namespace)
        self.assertEqual(
            infer(None, object(), ids, weights),
            "cpu-after-query-failure",
        )
        self.assertEqual(fetch.calls, [(17, [3, 7, 9], False)])
        self.assertEqual(len(cpu_calls), 1)

    def test_slow_reference_moves_weights_to_activation_device(self):
        moves = []
        installed = []

        class Weight:
            def __init__(self, name):
                self.name = name

            def to(self, *, device, dtype):
                moves.append((self.name, device, dtype))
                return ("moved", self.name, device, dtype)

        namespace = {
            "set_param": lambda expert, name, value: installed.append(
                (expert, name, value)
            ),
            "_orig_moe_infer": lambda *_args: "reference-result",
            "torch": torch,
        }
        infer = extract_driver_function("_slow_moe_infer", namespace)
        experts = {5: object(), 8: object()}
        owner = types.SimpleNamespace(experts=experts)
        raw = {
            expert: {
                name: Weight(f"{expert}.{name}")
                for name in ("w1", "w2", "w3")
            }
            for expert in experts
        }
        x = types.SimpleNamespace(
            device=torch.device("cuda:1"),
            dtype=torch.float16,
        )
        self.assertEqual(
            infer(owner, x, object(), object(), raw, [5, 8]),
            "reference-result",
        )
        self.assertEqual(len(moves), 6)
        self.assertTrue(all(
            device == torch.device("cuda:1") and dtype == torch.float16
            for _name, device, dtype in moves
        ))
        # Six installed weights followed by six meta releases.
        self.assertEqual(len(installed), 12)

    def test_slow_reference_cleans_partial_install_after_copy_failure(self):
        installed = []
        copy_calls = []

        class Weight:
            def to(self, *, device, dtype):
                copy_calls.append((device, dtype))
                if len(copy_calls) == 2:
                    raise RuntimeError("synthetic accelerator OOM")
                return self

        namespace = {
            "set_param": lambda expert, name, value: installed.append(
                (expert, name, value)
            ),
            "_orig_moe_infer": lambda *_args: self.fail(
                "reference inference ran after a failed weight copy"
            ),
            "torch": torch,
        }
        infer = extract_driver_function("_slow_moe_infer", namespace)
        experts = {5: object(), 8: object()}
        owner = types.SimpleNamespace(experts=experts)
        raw = {
            expert: {name: Weight() for name in ("w1", "w2", "w3")}
            for expert in experts
        }
        x = types.SimpleNamespace(
            device=torch.device("cuda:0"), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "synthetic accelerator OOM"):
            infer(owner, x, object(), object(), raw, [5, 8])
        released = [
            (expert, name)
            for expert, name, value in installed
            if isinstance(value, torch.Tensor) and value.device.type == "meta"
        ]
        self.assertEqual(
            released,
            [
                (experts[expert], f"{name}.weight")
                for expert in (5, 8)
                for name in ("w1", "w2", "w3")
            ],
        )

    def test_cuda_startup_respects_kill_switch_and_dtype_gate(self):
        available_calls = []
        fake_cuda = types.SimpleNamespace(
            available=lambda device: (
                available_calls.append(device) or True
            ),
            describe=lambda _device: "qualified fake CUDA",
            moe_infer=object(),
        )

        def run(*, fast, dtype):
            disabled = []
            namespace = {
                "cuda_moe": None,
                "_MOE_FN": object(),
                "_CUDA_MOE_ACTIVE": False,
                "FAST_MOE": fast,
                "_disable_cuda_moe": disabled.append,
                "DEV": types.SimpleNamespace(type="cuda"),
                "DT": dtype,
                "torch": torch,
                "print": lambda *_args, **_kwargs: None,
            }
            activate = extract_driver_function(
                "_activate_cuda_moe", namespace)
            with mock.patch.dict(sys.modules, {"cuda_moe": fake_cuda}):
                activate()
            return namespace, disabled

        namespace, disabled = run(fast=False, dtype=torch.float32)
        self.assertEqual(disabled, ["K3_FAST_MOE=0"])
        self.assertFalse(namespace["_CUDA_MOE_ACTIVE"])
        self.assertEqual(available_calls, [])

        namespace, disabled = run(fast=True, dtype=torch.float16)
        self.assertEqual(len(disabled), 1)
        self.assertIn("requires fp32", disabled[0])
        self.assertFalse(namespace["_CUDA_MOE_ACTIVE"])
        self.assertEqual(available_calls, [])

        namespace, disabled = run(fast=True, dtype=torch.float32)
        self.assertEqual(disabled, [])
        self.assertTrue(namespace["_CUDA_MOE_ACTIVE"])
        self.assertIs(namespace["_MOE_FN"], fake_cuda.moe_infer)
        self.assertEqual(len(available_calls), 1)

    def test_disable_cuda_restores_cpu_backend_state(self):
        disabled = []
        messages = []
        cpu_backend = object()
        namespace = {
            "_CUDA_MOE_ACTIVE": True,
            "_CUDA_MOE_FALLBACK_REPORTED": False,
            "MOE_BACKEND": "cuda",
            "_MOE_FN": object(),
            "FAST_MOE": True,
            "_CPU_MOE_FN": cpu_backend,
            "cuda_moe": types.SimpleNamespace(
                disable=lambda device, reason: disabled.append(
                    (device, reason)
                )
            ),
            "DEV": torch.device("cuda:2"),
            "print": lambda message, **_kwargs: messages.append(message),
        }
        disable = extract_driver_function("_disable_cuda_moe", namespace)
        disable("synthetic failure")
        self.assertFalse(namespace["_CUDA_MOE_ACTIVE"])
        self.assertEqual(namespace["MOE_BACKEND"], "cpu")
        self.assertIs(namespace["_MOE_FN"], cpu_backend)
        self.assertTrue(namespace["FAST_MOE"])
        self.assertEqual(
            disabled,
            [(torch.device("cuda:2"), "synthetic failure")],
        )
        self.assertEqual(len(messages), 1)

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
