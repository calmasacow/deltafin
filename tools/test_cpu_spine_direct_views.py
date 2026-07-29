#!/usr/bin/env python3
"""Focused C55 gates for CPU direct resident-spine buffer views.

These tests use tiny synthetic packs.  They never read model weights, compile
Metal, initialize CUDA/ROCm, or execute native expert code.
"""

from __future__ import annotations

import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mixed_spine  # noqa: E402
import spine_codec  # noqa: E402
import spine_fast  # noqa: E402


def tensor_bytes(tensor):
    return bytearray(
        tensor.contiguous().view(torch.uint8).reshape(-1).tolist()
    )


class TinyLayer(torch.nn.Module):
    def __init__(self, shape, tail_size=4):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full(shape, float("nan"), dtype=torch.float32),
            requires_grad=False,
        )
        self.tail = torch.nn.Parameter(
            torch.full(
                (tail_size,), float("nan"), dtype=torch.float32
            ),
            requires_grad=False,
        )


def forbidden(*_args, **_kwargs):
    raise AssertionError("cold fallback unexpectedly executed")


def spine_pack(q_values, scales, tail):
    q_buffer = tensor_bytes(q_values)
    scale_buffer = tensor_bytes(scales)
    tail_buffer = tensor_bytes(tail)
    rows, cols = q_values.shape
    layout = {
        "items": [
            (
                "weight",
                "layer.weight",
                (rows, cols),
                0,
                q_values.numel(),
                0,
                rows,
            )
        ],
        "qtotal": q_values.numel(),
        "sctotal": rows,
        "oplan": [
            (
                "tail",
                "layer.tail",
                "BF16",
                (tail.numel(),),
                0,
                tail.numel() * tail.element_size(),
                None,
            )
        ],
        "ototal": tail.numel() * tail.element_size(),
        "other": [("tail", "layer.tail", None)],
        "int8_dir": "/unused",
    }
    return (
        {
            "lay": layout,
            "q": q_buffer,
            "sc": scale_buffer,
            "other": tail_buffer,
        },
        q_buffer,
        scale_buffer,
        tail_buffer,
    )


def mixed_pack(source, codec, tail):
    packed, scales, dequant = spine_codec.quantize(
        source, codec, 32
    )
    q_buffer = bytearray(packed.tobytes())
    scale_buffer = bytearray(scales.tobytes())
    tail_buffer = tensor_bytes(tail)
    rows, cols = source.shape
    layout = {
        "items": [
            (
                "weight",
                "layer.weight",
                codec,
                "/unused",
                (rows, cols),
                0,
                packed.nbytes,
                0,
                scales.nbytes,
            )
        ],
        "qtotal": packed.nbytes,
        "stotal": scales.size,
        "oplan": [
            (
                "tail",
                "layer.tail",
                "BF16",
                (tail.numel(),),
                0,
                tail.numel() * tail.element_size(),
                None,
            )
        ],
        "ototal": tail.numel() * tail.element_size(),
        "other": [("tail", "layer.tail", None)],
        "g": 32,
    }
    return (
        {
            "lay": layout,
            "q": q_buffer,
            "sc": scale_buffer,
            "other": tail_buffer,
        },
        q_buffer,
        scale_buffer,
        tail_buffer,
        torch.from_numpy(dequant),
    )


class FakePackedStage:
    storage_mode = "stage"
    stage_sync_mode = "event"
    enabled = True

    def __init__(self):
        self.bound_q = None
        self.owned_scales = None
        self.q_input_ptr = None
        self.scale_input_ptr = None
        self.started = False
        self.finished = False

    def begin_load(self):
        self.started = True

    def stage_descriptor(self, _items):
        return {"synthetic": True}

    def bind_stage(self, qweight, scales, _descriptor, _lease):
        self.bound_q = qweight
        self.owned_scales = scales.clone()
        self.q_input_ptr = qweight.data_ptr()
        self.scale_input_ptr = scales.data_ptr()
        return True

    def load(self, name, _qweight, _scale):
        return name == "weight"

    def consumes(self, _name):
        return False

    def mark_dense(self, _name):
        raise AssertionError("packed synthetic weight became dense")

    def finish_load(self):
        self.finished = True


class CpuSpineDirectViewTests(unittest.TestCase):
    def test_branch_selection_is_runtime_capability_gated(self):
        cpu = types.SimpleNamespace(type="cpu")
        arena = types.SimpleNamespace(storage_mode="arena")
        stage = types.SimpleNamespace(storage_mode="stage")
        self.assertEqual(
            spine_fast._cpu_direct_view_flags(cpu), (True, True, True)
        )
        self.assertEqual(
            spine_fast._cpu_direct_view_flags(cpu, arena),
            (True, True, True),
        )
        self.assertEqual(
            spine_fast._cpu_direct_view_flags(cpu, stage),
            (False, True, True),
        )
        self.assertTrue(mixed_spine._cpu_direct_views(cpu))

        # CUDA is also PyTorch's ROCm device spelling.  Literal future
        # spellings and unknown accelerators must fail closed too.
        for device_type in (
            "mps",
            "cuda",
            "rocm",
            "hip",
            "xpu",
            "privateuseone",
            None,
        ):
            with self.subTest(device_type=device_type):
                dev = types.SimpleNamespace(type=device_type)
                self.assertEqual(
                    spine_fast._cpu_direct_view_flags(dev),
                    (False, False, False),
                )
                self.assertFalse(mixed_spine._cpu_direct_views(dev))

    def test_int8_cpu_apply_is_exact_and_owns_materialized_values(self):
        q_values = torch.tensor(
            [
                [-127, -31, -1, 0, 1, 31, 63, 127],
                [17, -19, 23, -29, 31, -37, 41, -43],
            ],
            dtype=torch.int8,
        )
        scales = torch.tensor([2**-8, 0.03125], dtype=torch.float16)
        tail = torch.tensor(
            [0.0, 1.0, -2.0, 0.125], dtype=torch.bfloat16
        )
        pack, q_buffer, scale_buffer, tail_buffer = spine_pack(
            q_values, scales, tail
        )
        layer = TinyLayer(q_values.shape, tail.numel())
        expected_weight = (
            q_values.to(torch.float32)
            * scales.to(torch.float32).view(-1, 1)
        )
        expected_tail = tail.to(torch.float32)

        h2d_before = spine_fast.PHASE["h2d_bytes"]
        with (
            mock.patch.object(
                spine_fast,
                "_stage_weight_buf",
                side_effect=AssertionError("CPU q staging was used"),
            ),
            mock.patch.object(
                spine_fast,
                "_stage_buf",
                side_effect=AssertionError("CPU scale/tail staging was used"),
            ),
            mock.patch.object(
                spine_fast, "dynamic_q8_qkv", return_value=None
            ),
            mock.patch.object(spine_fast, "_release"),
            mock.patch.object(spine_fast, "DEQ", "torch"),
        ):
            spine_fast.apply_pack(
                layer,
                "layer.",
                pack,
                torch.device("cpu"),
                torch.float32,
                {},
                {"BF16": torch.bfloat16},
                forbidden,
                forbidden,
            )

        self.assertTrue(torch.equal(layer.weight, expected_weight))
        self.assertTrue(torch.equal(layer.tail, expected_tail))
        self.assertEqual(spine_fast.PHASE["h2d_bytes"], h2d_before)
        weight_before = layer.weight.detach().clone()
        tail_before = layer.tail.detach().clone()
        q_buffer[0] ^= 0x7F
        scale_buffer[0] ^= 0x7F
        tail_buffer[0] ^= 0x7F
        self.assertTrue(torch.equal(layer.weight, weight_before))
        self.assertTrue(torch.equal(layer.tail, tail_before))

    def test_mixed_cpu_apply_is_exact_for_every_codec_and_owns_values(self):
        rng = np.random.default_rng(20260755)
        source = rng.standard_normal((5, 70), dtype=np.float32) * 0.03
        source[0, :8] = np.array(
            [0.0, 0.5, -0.5, 0.125, -0.125, 1.0, -1.0, 0.0],
            dtype=np.float32,
        )
        tail = torch.tensor(
            [0.0, 1.0, -2.0, 0.125], dtype=torch.bfloat16
        )
        for codec in spine_codec.CODECS:
            with self.subTest(codec=codec):
                (
                    pack,
                    q_buffer,
                    scale_buffer,
                    tail_buffer,
                    expected_weight,
                ) = mixed_pack(source, codec, tail)
                layer = TinyLayer(source.shape, tail.numel())
                h2d_before = mixed_spine.PHASE["h2d_bytes"]
                with (
                    mock.patch.object(
                        mixed_spine,
                        "_stage_buf",
                        side_effect=AssertionError(
                            "mixed CPU staging was used"
                        ),
                    ),
                    mock.patch.object(mixed_spine, "_release"),
                    mock.patch.object(mixed_spine, "DEQ", "torch"),
                ):
                    mixed_spine.apply_pack(
                        layer,
                        "layer.",
                        pack,
                        torch.device("cpu"),
                        torch.float32,
                        {},
                        {"BF16": torch.bfloat16},
                        forbidden,
                        forbidden,
                    )

                self.assertTrue(torch.equal(layer.weight, expected_weight))
                self.assertTrue(
                    torch.equal(layer.tail, tail.to(torch.float32))
                )
                self.assertEqual(
                    mixed_spine.PHASE["h2d_bytes"], h2d_before
                )
                weight_before = layer.weight.detach().clone()
                tail_before = layer.tail.detach().clone()
                q_buffer[0] ^= 0x7F
                scale_buffer[0] ^= 0x7F
                tail_buffer[0] ^= 0x7F
                self.assertTrue(torch.equal(layer.weight, weight_before))
                self.assertTrue(torch.equal(layer.tail, tail_before))

    def test_packed_stage_retains_owned_q_but_uses_direct_scale_and_tail(self):
        q_values = torch.tensor(
            [[-7, -3, -1, 0, 1, 3, 5, 7]], dtype=torch.int8
        )
        scales = torch.tensor([0.125], dtype=torch.float16)
        tail = torch.tensor(
            [0.0, 1.0, -2.0, 0.125], dtype=torch.bfloat16
        )
        pack, q_buffer, scale_buffer, tail_buffer = spine_pack(
            q_values, scales, tail
        )
        layer = TinyLayer(q_values.shape, tail.numel())
        controller = FakePackedStage()
        stage_calls = []
        h2d_before = spine_fast.PHASE["h2d_bytes"]

        def make_q_stage(n, dev, sync_mode):
            stage = torch.empty(n, dtype=torch.int8, device=dev)
            stage_calls.append((n, dev.type, sync_mode, stage))
            return stage, object()

        with (
            mock.patch.object(
                spine_fast,
                "dynamic_q8_qkv",
                return_value=controller,
            ),
            mock.patch.object(
                spine_fast,
                "_stage_weight_buf",
                side_effect=make_q_stage,
            ),
            mock.patch.object(
                spine_fast,
                "_stage_buf",
                side_effect=AssertionError(
                    "packed-stage CPU scale/tail staging was used"
                ),
            ),
            mock.patch.object(spine_fast, "_release"),
            mock.patch.object(spine_fast, "DEQ", "torch"),
        ):
            spine_fast.apply_pack(
                layer,
                "layer.",
                pack,
                torch.device("cpu"),
                torch.float32,
                {},
                {"BF16": torch.bfloat16},
                forbidden,
                forbidden,
            )

        self.assertTrue(controller.started)
        self.assertTrue(controller.finished)
        self.assertEqual(len(stage_calls), 1)
        self.assertEqual(
            spine_fast.PHASE["h2d_bytes"] - h2d_before,
            q_values.numel(),
        )
        self.assertTrue(torch.equal(controller.bound_q, q_values.view(-1)))
        self.assertTrue(
            torch.equal(controller.owned_scales, scales.view(-1))
        )
        self.assertNotEqual(
            controller.q_input_ptr,
            torch.frombuffer(q_buffer, dtype=torch.int8).data_ptr(),
        )
        self.assertEqual(
            controller.scale_input_ptr,
            torch.frombuffer(
                scale_buffer, dtype=torch.float16
            ).data_ptr(),
        )
        self.assertTrue(torch.equal(layer.tail, tail.to(torch.float32)))

        q_before = controller.bound_q.clone()
        scales_before = controller.owned_scales.clone()
        tail_before = layer.tail.detach().clone()
        q_buffer[0] ^= 0x7F
        scale_buffer[0] ^= 0x7F
        tail_buffer[0] ^= 0x7F
        self.assertTrue(torch.equal(controller.bound_q, q_before))
        self.assertTrue(
            torch.equal(controller.owned_scales, scales_before)
        )
        self.assertTrue(torch.equal(layer.tail, tail_before))


if __name__ == "__main__":
    unittest.main()
