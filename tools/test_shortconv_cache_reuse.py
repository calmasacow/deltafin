#!/usr/bin/env python3
"""Regression gate for allocation-free ShortConvolution cache extraction."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fla.modules as modules  # noqa: E402


class ShortConvCacheReuseTests(unittest.TestCase):
    def setUp(self):
        self.old_mode = modules.SHORTCONV
        self.old_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        modules.SHORTCONV = self.old_mode
        torch.set_grad_enabled(self.old_grad_enabled)

    def test_each_backend_formula_builds_one_source_and_preserves_cache(self):
        generator = torch.Generator().manual_seed(52_052)
        for mode in ("mulsum", "conv1d"):
            modules.SHORTCONV = mode
            for batch in (1, 2):
                for tokens in (1, 2, 8, 9):
                    for cache_present in (False, True):
                        with self.subTest(
                            mode=mode,
                            batch=batch,
                            tokens=tokens,
                            cache_present=cache_present,
                        ):
                            channels, width = 17, 4
                            module = modules.ShortConvolution(
                                channels,
                                width,
                                bias=False,
                                activation="silu",
                            ).eval()
                            with torch.no_grad():
                                module.weight.copy_(
                                    torch.randn(
                                        module.weight.shape,
                                        generator=generator,
                                    )
                                )
                            x = torch.randn(
                                batch,
                                tokens,
                                channels,
                                generator=generator,
                            )
                            residual = torch.randn(
                                batch,
                                tokens,
                                channels,
                                generator=generator,
                            )
                            cache = (
                                torch.randn(
                                    batch,
                                    channels,
                                    width,
                                    generator=generator,
                                )
                                if cache_present
                                else None
                            )
                            old_cache = (
                                None if cache is None else cache.clone()
                            )
                            expected_prefix = (
                                cache.float()
                                if cache is not None
                                else x.new_zeros(
                                    batch, channels, width
                                ).float()
                            )
                            expected_cache = torch.cat(
                                (expected_prefix, x.transpose(1, 2).float()),
                                dim=-1,
                            )[:, :, -width:].contiguous()

                            real_cat = torch.cat
                            sources = []

                            def recording_cat(*args, **kwargs):
                                result = real_cat(*args, **kwargs)
                                sources.append(result)
                                return result

                            with (
                                mock.patch.object(
                                    modules.torch,
                                    "cat",
                                    side_effect=recording_cat,
                                ),
                                torch.inference_mode(),
                            ):
                                output, got_cache = module(
                                    x,
                                    residual=residual,
                                    cache=cache,
                                    output_final_state=True,
                                )

                            self.assertFalse(torch.is_grad_enabled())
                            self.assertEqual(len(sources), 1)
                            self.assertTrue(
                                torch.equal(got_cache, expected_cache)
                            )
                            self.assertEqual(
                                got_cache.dtype, torch.float32
                            )
                            self.assertEqual(
                                output.shape, (batch, tokens, channels)
                            )
                            if tokens == 1:
                                self.assertEqual(
                                    got_cache.untyped_storage().data_ptr(),
                                    sources[0].untyped_storage().data_ptr(),
                                )
                            if cache is not None:
                                self.assertTrue(
                                    torch.equal(cache, old_cache)
                                )
                                self.assertNotEqual(
                                    got_cache.untyped_storage().data_ptr(),
                                    cache.untyped_storage().data_ptr(),
                                )

    def test_old_returned_cache_remains_immutable_across_next_call(self):
        modules.SHORTCONV = "mulsum"
        module = modules.ShortConvolution(
            31, 4, bias=False, activation="silu"
        ).eval()
        generator = torch.Generator().manual_seed(520_053)
        with torch.no_grad():
            module.weight.copy_(
                torch.randn(module.weight.shape, generator=generator)
            )
        first_x = torch.randn(1, 1, 31, generator=generator)
        second_x = torch.randn(1, 1, 31, generator=generator)
        _, first_cache = module(
            first_x, cache=torch.zeros(1, 31, 4),
            output_final_state=True,
        )
        snapshot = first_cache.clone()
        pointer = first_cache.untyped_storage().data_ptr()
        _, second_cache = module(
            second_x, cache=first_cache, output_final_state=True
        )
        self.assertEqual(
            first_cache.untyped_storage().data_ptr(), pointer
        )
        self.assertTrue(torch.equal(first_cache, snapshot))
        self.assertNotEqual(
            second_cache.untyped_storage().data_ptr(), pointer
        )

    def test_grad_enabled_cache_mutation_does_not_poison_backward(self):
        """Training keeps the historical cache/source independence contract."""
        generator = torch.Generator().manual_seed(52_054)
        for mode in ("mulsum", "conv1d"):
            with self.subTest(mode=mode), torch.enable_grad():
                modules.SHORTCONV = mode
                module = modules.ShortConvolution(
                    13, 4, bias=False, activation="silu"
                ).eval()
                with torch.no_grad():
                    module.weight.copy_(
                        torch.randn(
                            module.weight.shape,
                            generator=generator,
                        )
                    )

                x = torch.randn(
                    1, 1, 13, generator=generator, requires_grad=True
                )
                cache = torch.randn(
                    1, 13, 4, generator=generator, requires_grad=True
                )
                clean_x = x.detach().clone().requires_grad_(True)
                clean_cache = cache.detach().clone().requires_grad_(True)

                output, returned_cache = module(
                    x,
                    cache=cache,
                    output_final_state=True,
                )
                clean_output, clean_returned_cache = module(
                    clean_x,
                    cache=clean_cache,
                    output_final_state=True,
                )
                self.assertTrue(
                    torch.equal(returned_cache, clean_returned_cache)
                )

                # This was legal before source-tail reuse. If returned_cache
                # aliases the causal source saved for backward, this mutation
                # trips a version-counter error in both arithmetic branches.
                with torch.no_grad():
                    returned_cache.add_(123.0)

                got_grads = torch.autograd.grad(
                    output.square().sum(),
                    (x, cache, module.weight),
                )
                expected_grads = torch.autograd.grad(
                    clean_output.square().sum(),
                    (clean_x, clean_cache, module.weight),
                )
                for got, expected in zip(got_grads, expected_grads):
                    torch.testing.assert_close(
                        got, expected, rtol=0.0, atol=0.0
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
