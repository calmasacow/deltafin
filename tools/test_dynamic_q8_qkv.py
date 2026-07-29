#!/usr/bin/env python3
"""Disk-free tests for the capability-gated dynamic packed-q8 KDA projections."""
import os
import sys
import unittest

import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(__file__))
import spine_fast  # noqa: E402


class _Attention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)


class _Layer(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.self_attn = _Attention(width)


def _reference_mm(hidden, qweight, scale):
    return hidden @ (qweight.float() * scale[:, None]).T


def _source_for(index, width=8):
    qweight = (
        torch.arange(width * width, dtype=torch.int16).reshape(width, width)
        .remainder(13).sub(6).to(torch.int8)
        .add(index)
    )
    scale = torch.linspace(
        0.125 * index, 0.25 * index, width, dtype=torch.float16)
    return qweight, scale


def _set_param(root, dotted, tensor):
    obj = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))


def _apply_sources(layer, packed_sources, dense_sources):
    """Apply a tiny synthetic spine pack through the production integration."""
    qbuf = bytearray()
    scbuf = bytearray()
    items = []
    for role, (qweight, scale) in packed_sources.items():
        name = f"self_attn.{role}_proj.weight"
        qoff = len(qbuf)
        scoff = len(scbuf) // 2
        qbuf.extend(
            qweight.contiguous().reshape(-1).view(torch.uint8).tolist())
        scbuf.extend(scale.contiguous().view(torch.uint8).tolist())
        rows, cols = qweight.shape
        items.append((
            name,
            "layer." + name,
            (rows, cols),
            qoff,
            rows * cols,
            scoff,
            rows,
        ))

    dense_by_full = {}
    other = []
    for role, dense in dense_sources.items():
        name = f"self_attn.{role}_proj.weight"
        full = "layer." + name
        dense_by_full[full] = dense
        other.append((name, full, None))

    pack = {
        "lay": {
            "items": items,
            "qtotal": len(qbuf),
            "sctotal": len(scbuf) // 2,
            "oplan": [],
            "ototal": 0,
            "other": other,
        },
        "q": qbuf,
        "sc": scbuf,
        "other": bytearray(1),
    }
    spine_fast.apply_pack(
        layer,
        "layer.",
        pack,
        torch.device("cpu"),
        torch.float32,
        {},
        {},
        _set_param,
        lambda full: dense_by_full[full],
    )


class DynamicQ8QKVTests(unittest.TestCase):
    def _controller(self, matmul=_reference_mm, meta=False):
        if meta:
            with torch.device("meta"):
                layer = _Layer()
        else:
            layer = _Layer()
        state = spine_fast.DynamicQ8State()
        controller = spine_fast.install_dynamic_q8_qkv(
            layer, torch.device("cpu"), state, matmul)
        return layer, state, controller

    @staticmethod
    def _load(controller):
        source = {}
        for index, role in enumerate(("q", "k", "v"), start=1):
            qweight, scale = _source_for(index)
            source[role] = (qweight, scale)
            consumed = controller.load(
                f"self_attn.{role}_proj.weight", qweight, scale)
            assert consumed
        controller.finish_load()
        return source

    def test_owned_arenas_survive_recycled_source_mutation(self):
        layer, state, controller = self._controller()
        controller.begin_load()
        source = self._load(controller)
        hidden = torch.randn(2, 3, 8)

        expected = {
            role: _reference_mm(hidden, qweight, scale.float())
            for role, (qweight, scale) in source.items()
        }
        for qweight, scale in source.values():
            qweight.zero_()
            scale.zero_()

        self.assertTrue(state.enabled)
        for role in ("q", "k", "v"):
            got = getattr(layer.self_attn, f"{role}_proj")(hidden)
            torch.testing.assert_close(got, expected[role], rtol=0, atol=0)
        self.assertEqual(controller.packed_project_calls, 3)

    def test_operator_error_materializes_dense_and_disables_every_template(self):
        calls = 0

        def failing_mm(hidden, qweight, scale):
            nonlocal calls
            calls += 1
            raise RuntimeError("synthetic backend rejection")

        layer, state, controller = self._controller(failing_mm)
        controller.begin_load()
        source = self._load(controller)
        hidden = torch.randn(1, 2, 8)

        got = layer.self_attn.q_proj(hidden)
        expected = _reference_mm(
            hidden, source["q"][0], source["q"][1].float())
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertEqual(calls, 1)
        self.assertFalse(state.enabled)
        self.assertIn("synthetic backend rejection", state.reason)

        # All current roles were reconstructed, not only the one that failed.
        for role in ("q", "k", "v"):
            qweight, scale = source[role]
            expected_weight = qweight.float() * scale.float()[:, None]
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj").weight,
                expected_weight,
                rtol=0,
                atol=0,
            )

        # Once disabled, future layer application must take the normal dense
        # branch instead of silently reusing an old packed layer.
        controller.begin_load()
        self.assertFalse(controller.load(
            "self_attn.q_proj.weight",
            torch.ones(8, 8, dtype=torch.int8),
            torch.ones(8, dtype=torch.float16),
        ))

    def test_incomplete_layer_fails_closed(self):
        layer, state, controller = self._controller()
        controller.begin_load()
        qweight = torch.ones(8, 8, dtype=torch.int8)
        scale = torch.full((8,), 0.5, dtype=torch.float16)
        self.assertTrue(controller.load(
            "self_attn.q_proj.weight", qweight, scale))
        with self.assertRaisesRegex(
                RuntimeError, "no packed or dense value for k, v"):
            controller.finish_load()

        self.assertFalse(state.enabled)
        self.assertIn("no packed or dense value for k, v", state.reason)
        torch.testing.assert_close(
            layer.self_attn.q_proj.weight,
            torch.full((8, 8), 0.5),
            rtol=0,
            atol=0,
        )

    def test_two_templates_can_share_persistent_arenas(self):
        first, state, controller_a = self._controller()
        second = _Layer()
        controller_b = spine_fast.install_dynamic_q8_qkv(
            second,
            torch.device("cpu"),
            state,
            _reference_mm,
            arenas=controller_a.arenas(),
        )

        self.assertEqual(
            controller_a.arenas()[0].data_ptr(),
            controller_b.arenas()[0].data_ptr(),
        )
        self.assertEqual(
            controller_a.arenas()[1].data_ptr(),
            controller_b.arenas()[1].data_ptr(),
        )

    def test_backend_error_replaces_meta_dense_fallback(self):
        def failing_mm(hidden, qweight, scale):
            raise NotImplementedError("synthetic missing kernel")

        layer, state, controller = self._controller(failing_mm, meta=True)
        controller.begin_load()
        source = self._load(controller)
        hidden = torch.randn(1, 8)

        got = layer.self_attn.v_proj(hidden)
        expected = _reference_mm(
            hidden, source["v"][0], source["v"][1].float())
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertFalse(state.enabled)
        for role in ("q", "k", "v"):
            self.assertEqual(
                getattr(layer.self_attn, f"{role}_proj").weight.device.type,
                "cpu",
            )

    def test_hybrid_packed_and_dense_layer_falls_back_fully_dense(self):
        layer, state, controller = self._controller(meta=True)
        sources = {
            role: _source_for(index)
            for index, role in enumerate(("q", "k", "v"), start=1)
        }
        dense_v = (
            sources["v"][0].float() * sources["v"][1].float()[:, None])

        _apply_sources(
            layer,
            {"q": sources["q"], "k": sources["k"]},
            {"v": dense_v},
        )

        self.assertFalse(state.enabled)
        self.assertIn("dense projection(s): v", state.reason)
        hidden = torch.randn(2, 3, 8)
        for role in ("q", "k", "v"):
            qweight, scale = sources[role]
            expected_weight = qweight.float() * scale.float()[:, None]
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj").weight,
                expected_weight,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj")(hidden),
                _reference_mm(hidden, qweight, scale.float()),
                rtol=0,
                atol=0,
            )
        self.assertEqual(controller.packed_project_calls, 0)

    def test_copy_failure_materializes_packed_prefix_then_finishes_dense(self):
        layer, state, controller = self._controller(meta=True)
        sources = {
            role: _source_for(index)
            for index, role in enumerate(("q", "k", "v"), start=1)
        }
        original_views = controller._views

        class RejectingCopy:
            shape = sources["k"][0].shape

            @staticmethod
            def copy_(_source):
                raise RuntimeError("synthetic packed copy failure")

        def rejecting_k_view(role):
            if role == "k":
                return RejectingCopy(), original_views(role)[1]
            return original_views(role)

        controller._views = rejecting_k_view
        _apply_sources(layer, sources, {})

        self.assertFalse(state.enabled)
        self.assertIn("synthetic packed copy failure", state.reason)
        hidden = torch.randn(1, 2, 8)
        for role in ("q", "k", "v"):
            qweight, scale = sources[role]
            expected_weight = qweight.float() * scale.float()[:, None]
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj").weight,
                expected_weight,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{role}_proj")(hidden),
                _reference_mm(hidden, qweight, scale.float()),
                rtol=0,
                atol=0,
            )
        self.assertEqual(controller.packed_project_calls, 0)

    def test_uninstall_restores_linear_forward_and_removes_buffers(self):
        layer, _state, controller = self._controller()
        controller.uninstall()

        self.assertNotIn("forward", layer.self_attn.q_proj.__dict__)
        self.assertIsNone(spine_fast.dynamic_q8_qkv(layer))
        self.assertNotIn(controller._QBUF, layer.self_attn._buffers)
        self.assertNotIn(controller._SBUF, layer.self_attn._buffers)


if __name__ == "__main__":
    unittest.main()
