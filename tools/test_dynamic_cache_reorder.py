#!/usr/bin/env python3
"""Focused cross-device regression tests for KimiDynamicCache beam reorder."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attn_fast
from k3pkg import modeling_kimi_linear as ml

attn_fast.install_cache_portability(ml)


class CacheConfig:
    linear_attn_config = None
    num_hidden_layers = 2


class PristineCache:
    """The public setup shape, without local geometric-slab extensions."""

    def __init__(self, _config):
        self.recurrent_states = [None, None]
        self.conv_states = [None, None]
        self.key_cache = [None, None]
        self.value_cache = [None, None]

    def reorder_cache(self, _beam):
        raise AssertionError("portable reorder patch was not installed")


class BeamProbe:
    def __init__(self, values: tuple[int, ...]):
        self.tensor = torch.tensor(values, dtype=torch.long)
        self.to_calls: list[tuple[torch.device, torch.dtype]] = []

    def to(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        self.to_calls.append((device, dtype))
        return self.tensor.to(device=device, dtype=dtype)


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(getattr(backend, "is_available", lambda: False)())


def rows(
    shape: tuple[int, ...],
    device: torch.device,
    offset: int,
) -> torch.Tensor:
    return (
        torch.arange(
            offset,
            offset + torch.tensor(shape).prod().item(),
            dtype=torch.float32,
            device=device,
        )
        .reshape(shape)
        .contiguous()
    )


class DynamicCacheReorderTests(unittest.TestCase):
    beam_rows = (2, 0, 2, 1)

    def exercise(
        self,
        compute_device: torch.device,
        recurrent_device: torch.device,
        beam_device: torch.device,
        cache_class=ml.KimiDynamicCache,
    ) -> None:
        cache = cache_class(CacheConfig())
        cache.recurrent_states[0] = rows(
            (3, 2, 2, 2), recurrent_device, 0
        )
        cache.conv_states[0] = tuple(
            rows((3, 4, 3), compute_device, 100 * (slot + 1))
            for slot in range(3)
        )
        key = rows((3, 2, 5, 4), compute_device, 500)
        value = rows((3, 2, 5, 3), compute_device, 1000)
        cache.key_cache[1] = key
        cache.value_cache[1] = value
        has_slab_bookkeeping = all(hasattr(cache, name) for name in (
            "_key_storage", "_value_storage", "_kv_capacity"
        ))
        if has_slab_bookkeeping:
            cache._key_storage[1] = key
            cache._value_storage[1] = value
            cache._kv_capacity[1] = key.shape[2]

        before = {
            "recurrent": cache.recurrent_states[0].clone(),
            "q_conv": cache.conv_states[0][0].clone(),
            "k_conv": cache.conv_states[0][1].clone(),
            "v_conv": cache.conv_states[0][2].clone(),
            "key": key.clone(),
            "value": value.clone(),
        }
        source = {
            "recurrent": cache.recurrent_states[0],
            "q_conv": cache.conv_states[0][0],
            "k_conv": cache.conv_states[0][1],
            "v_conv": cache.conv_states[0][2],
            "key": key,
            "value": value,
        }
        beam = torch.tensor(
            self.beam_rows, dtype=torch.long, device=beam_device
        )
        beam_before = beam.clone()
        cache.reorder_cache(beam)

        after = {
            "recurrent": cache.recurrent_states[0],
            "q_conv": cache.conv_states[0][0],
            "k_conv": cache.conv_states[0][1],
            "v_conv": cache.conv_states[0][2],
            "key": cache.key_cache[1],
            "value": cache.value_cache[1],
        }
        for name, tensor in after.items():
            expected = before[name].index_select(
                0, beam.to(before[name].device)
            )
            self.assertTrue(
                torch.equal(tensor, expected),
                f"{name} row order differs",
            )
            self.assertNotEqual(
                tensor.untyped_storage().data_ptr(),
                source[name].untyped_storage().data_ptr(),
                f"{name} unexpectedly aliases its source",
            )
            self.assertTrue(
                torch.equal(source[name], before[name]),
                f"{name} source was mutated",
            )
        self.assertTrue(torch.equal(beam, beam_before))
        if has_slab_bookkeeping:
            self.assertIs(cache._key_storage[1], cache.key_cache[1])
            self.assertIs(cache._value_storage[1], cache.value_cache[1])
            self.assertEqual(cache._kv_capacity[1], 5)

    def test_cpu_same_device_duplicate_reorder(self) -> None:
        cpu = torch.device("cpu")
        self.exercise(cpu, cpu, cpu)

    def test_pristine_cache_without_slab_bookkeeping(self) -> None:
        namespace = SimpleNamespace(KimiDynamicCache=PristineCache)
        self.assertTrue(attn_fast.install_cache_portability(namespace))
        installed = PristineCache.reorder_cache
        self.assertTrue(attn_fast.install_cache_portability(namespace))
        self.assertIs(PristineCache.reorder_cache, installed)
        cpu = torch.device("cpu")
        self.exercise(cpu, cpu, cpu, cache_class=PristineCache)

    def test_beam_indices_convert_once_per_device(self) -> None:
        attn_fast.install_cache_portability(
            SimpleNamespace(KimiDynamicCache=PristineCache)
        )
        cache = PristineCache(CacheConfig())
        cpu = torch.device("cpu")
        cache.recurrent_states[0] = rows((3, 1), cpu, 0)
        cache.conv_states[0] = tuple(
            rows((3, 1), cpu, 10 + slot) for slot in range(3)
        )
        cache.key_cache[1] = rows((3, 1, 1), cpu, 20)
        cache.value_cache[1] = rows((3, 1, 1), cpu, 30)
        beam = BeamProbe(self.beam_rows)
        cache.reorder_cache(beam)
        self.assertEqual(beam.to_calls, [(cpu, torch.long)])

    def test_mps_same_device_duplicate_reorder(self) -> None:
        if not mps_available():
            self.skipTest("MPS is unavailable")
        mps = torch.device("mps")
        self.exercise(mps, mps, mps)

    def test_mps_compute_with_cpu_recurrent_duplicate_reorder(self) -> None:
        if not mps_available():
            self.skipTest("MPS is unavailable")
        self.exercise(
            torch.device("mps"),
            torch.device("cpu"),
            torch.device("mps"),
        )

    def test_cuda_or_rocm_same_device_duplicate_reorder(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm is unavailable")
        accelerator = torch.device("cuda", torch.cuda.current_device())
        self.exercise(accelerator, accelerator, accelerator)


if __name__ == "__main__":
    unittest.main()
