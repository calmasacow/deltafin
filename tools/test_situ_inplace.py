#!/usr/bin/env python3
"""Exactness and ownership tests for the allocation-light NumPy SiTU path."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import fast_moe_batch
except ImportError:
    from . import fast_moe_batch


def _bits(value):
    return np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)


class SituInplaceTests(unittest.TestCase):
    def test_randomized_bit_exact_and_guarded(self):
        rng = np.random.default_rng(20260729)
        for shape in ((1, 1), (1, 3072), (16, 3072), (3, 17)):
            gate = np.ascontiguousarray(
                rng.standard_normal(shape, dtype=np.float32))
            up = np.ascontiguousarray(
                rng.standard_normal(shape, dtype=np.float32))
            expected = fast_moe_batch._situ(gate, up)

            guard = np.full(gate.size + 34, np.float32(12345.5))
            scratch = guard[17:-17].reshape(shape)
            got_gate = gate.copy()
            got_up = up.copy()
            returned = fast_moe_batch._situ_inplace(
                got_gate, got_up, scratch)

            self.assertIs(returned, got_gate)
            self.assertTrue(np.array_equal(_bits(returned), _bits(expected)))
            self.assertTrue(np.all(guard[:17] == np.float32(12345.5)))
            self.assertTrue(np.all(guard[-17:] == np.float32(12345.5)))

    def test_model_range_extremes_are_bit_exact(self):
        gate = np.array(
            [-100.0, -30.0, -10.0, -4.0, -0.0, 0.0, 1e-7,
             4.0, 10.0, 30.0, 100.0],
            dtype=np.float32,
        )
        up = np.array(
            [100.0, -100.0, 30.0, -30.0, 0.0, -0.0, 1e-7,
             25.0, -25.0, 4.0, -4.0],
            dtype=np.float32,
        )
        with np.errstate(over="ignore"):
            expected = fast_moe_batch._situ(gate, up)
            got = fast_moe_batch._situ_inplace(
                gate.copy(), up.copy(), np.empty_like(gate))
        self.assertTrue(np.array_equal(_bits(got), _bits(expected)))

    def test_output_prefix_is_valid_contiguous_scratch(self):
        # The production K3 shape reuses the not-yet-written d_model output.
        ne, d_ff, d_model = 16, 3072, 7168
        output = np.empty((ne, d_model), dtype=np.float32)
        scratch = output.reshape(-1)[:ne * d_ff].reshape(ne, d_ff)
        self.assertTrue(scratch.flags.c_contiguous)
        self.assertTrue(np.shares_memory(scratch, output))
        self.assertEqual(scratch.nbytes, ne * d_ff * 4)

    def test_disposable_expert_rows_combine_bit_exact_in_place(self):
        rng = np.random.default_rng(2026072901)
        for top_k, d_model in ((1, 1), (3, 17), (16, 7168)):
            rows = np.ascontiguousarray(
                rng.standard_normal((top_k, d_model), dtype=np.float32))
            weights = rng.standard_normal(top_k, dtype=np.float32)
            expected = np.zeros(d_model, dtype=np.float32)
            for i, weight in enumerate(weights):
                expected += np.float32(weight) * rows[i]

            disposable = rows.copy()
            got = np.zeros(d_model, dtype=np.float32)
            for i, weight in enumerate(weights):
                np.multiply(
                    disposable[i], np.float32(weight), out=disposable[i])
                np.add(got, disposable[i], out=got)
            self.assertTrue(np.array_equal(_bits(got), _bits(expected)))


if __name__ == "__main__":
    unittest.main()
