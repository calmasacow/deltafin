#!/usr/bin/env python3
"""Regression tests for speculative emission bounds and capture lifetime."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr  # noqa: E402


def _logits(token: int, positions: int = 1, vocab: int = 64):
    vocab = max(vocab, token + 1)
    result = torch.zeros(1, positions, vocab)
    result[..., token] = 1
    return result


class GenerateBoundsTests(unittest.TestCase):
    def _run(self, burst, *, max_new=3, first=1):
        streamed = []
        logged = []
        with (
            mock.patch.object(
                kr, "forward_pass", return_value=_logits(first)
            ) as forward,
            mock.patch.object(
                kr, "_spec_step_deep", return_value=(list(burst), " spec-test")
            ),
            mock.patch.object(kr.spec_decode, "enabled", return_value=True),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=max_new,
                spec=True,
                on_token=streamed.append,
                log=lambda step, tag, start, tokens: logged.append(tokens),
            )
        return generated, streamed, logged, forward.call_count

    def test_deep_burst_cannot_exceed_max_new(self):
        generated, streamed, logged, calls = self._run(
            range(2, 10), max_new=3
        )
        self.assertEqual(generated, [1, 2, 3])
        self.assertEqual(streamed, generated)
        self.assertEqual(logged, [generated])
        self.assertEqual(calls, 1)

    def test_eos_trims_burst_before_streaming(self):
        generated, streamed, logged, _ = self._run(
            [2, kr.EOS_ID, 3, 4], max_new=10
        )
        self.assertEqual(generated, [1, 2, kr.EOS_ID])
        self.assertEqual(streamed, generated)
        self.assertEqual(logged, [generated])

    def test_zero_budget_does_not_run_prefill(self):
        generated, streamed, logged, calls = self._run(
            [2, 3], max_new=0
        )
        self.assertEqual(generated, [])
        self.assertEqual(streamed, [])
        self.assertEqual(logged, [])
        self.assertEqual(calls, 0)

    def test_prefill_eos_stops_immediately(self):
        generated, streamed, logged, calls = self._run(
            [2, 3], max_new=10, first=kr.EOS_ID
        )
        self.assertEqual(generated, [kr.EOS_ID])
        self.assertEqual(streamed, generated)
        self.assertEqual(logged, [])
        self.assertEqual(calls, 1)


class _ArgmaxVector:
    def __init__(self, values, counters):
        self._values = list(values)
        self._counters = counters

    def tolist(self):
        self._counters["bulk_transfers"] += 1
        return list(self._values)


class _VerifierRows:
    def __init__(self, values, counters):
        self._values = values
        self._counters = counters

    def argmax(self, dim):
        if dim != -1:
            raise AssertionError(f"unexpected argmax dim {dim}")
        self._counters["row_argmax_calls"] += 1
        return _ArgmaxVector(self._values, self._counters)


class _VerifierLogits:
    """Reject scalar-position reads so the test enforces one bulk handoff."""

    def __init__(self, values, counters):
        self._values = values
        self._counters = counters

    def __getitem__(self, index):
        if index != 0:
            self._counters["scalar_position_reads"] += 1
            raise AssertionError(f"unexpected scalar verifier read {index!r}")
        return _VerifierRows(self._values, self._counters)


class ShallowSpecArgmaxTests(unittest.TestCase):
    def test_accept_reads_both_argmax_ids_in_one_bulk_transfer(self):
        counters = {
            "row_argmax_calls": 0,
            "bulk_transfers": 0,
            "scalar_position_reads": 0,
        }
        verifier = _VerifierLogits([7, 8], counters)
        streamed = []
        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), verifier],
            ),
            mock.patch.object(kr, "ngram_draft", return_value=7),
            mock.patch.object(kr, "snapshot_states", return_value={}),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(kr.spec_decode, "enabled", return_value=False),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=3,
                spec=True,
                on_token=streamed.append,
            )
        self.assertEqual(generated, [1, 7, 8])
        self.assertEqual(streamed, generated)
        self.assertEqual(
            counters,
            {
                "row_argmax_calls": 1,
                "bulk_transfers": 1,
                "scalar_position_reads": 0,
            },
        )
        restore.assert_not_called()

    def test_miss_uses_first_bulk_id_then_restores_and_reruns(self):
        counters = {
            "row_argmax_calls": 0,
            "bulk_transfers": 0,
            "scalar_position_reads": 0,
        }
        verifier = _VerifierLogits([9, 63], counters)
        streamed = []
        snapshot = object()
        with (
            mock.patch.object(
                kr,
                "forward_pass",
                side_effect=[_logits(1), verifier, _logits(8)],
            ),
            mock.patch.object(kr, "ngram_draft", return_value=7),
            mock.patch.object(
                kr, "snapshot_states", return_value=snapshot
            ),
            mock.patch.object(kr, "restore_states") as restore,
            mock.patch.object(kr.spec_decode, "enabled", return_value=False),
            mock.patch.object(kr, "prefetch_prev_token"),
            mock.patch.dict(os.environ, {"K3_PREFETCH": "0"}),
        ):
            generated = kr.generate(
                [],
                object(),
                lambda ids: ids,
                [41],
                max_new=2,
                spec=True,
                on_token=streamed.append,
            )
        self.assertEqual(generated, [1, 8])
        self.assertEqual(streamed, generated)
        self.assertEqual(counters["row_argmax_calls"], 1)
        self.assertEqual(counters["bulk_transfers"], 1)
        self.assertEqual(counters["scalar_position_reads"], 0)
        restore.assert_called_once_with(mock.ANY, snapshot)


class ReplayCaptureLifetimeTests(unittest.TestCase):
    def _step(self, rollback):
        logits = torch.zeros(1, 3, 32)
        logits[0, 0, 7] = 1
        logits[0, 1, 8] = 1
        logits[0, 2, 9] = 1
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", rollback),
            mock.patch.object(kr.spec_decode, "next_depth", return_value=2),
            mock.patch.object(kr.spec_decode, "draft", return_value=[7, 8]),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm") as arm,
            mock.patch.object(kr.spec_decode, "release") as release,
            mock.patch.object(kr.spec_decode, "record"),
            mock.patch.object(kr, "snapshot_states", return_value={}),
            mock.patch.object(kr, "forward_pass", return_value=logits),
        ):
            new, _ = kr._spec_step_deep(
                [], object(), lambda ids: ids, [1, 2], 2, 1
            )
        return new, arm.call_count, release.call_count

    def test_replay_arms_and_releases_once(self):
        new, arms, releases = self._step("replay")
        self.assertEqual(new, [7, 8, 9])
        self.assertEqual((arms, releases), (1, 1))

    def test_rerun_never_arms_unused_capture(self):
        new, arms, releases = self._step("rerun")
        self.assertEqual(new, [7, 8, 9])
        self.assertEqual((arms, releases), (0, 0))

    def test_replay_releases_capture_after_forward_failure(self):
        with (
            mock.patch.object(kr.spec_decode, "ROLLBACK", "replay"),
            mock.patch.object(kr.spec_decode, "next_depth", return_value=2),
            mock.patch.object(kr.spec_decode, "draft", return_value=[7, 8]),
            mock.patch.object(kr.spec_decode, "snapshot_mla", return_value={}),
            mock.patch.object(kr.spec_decode, "arm") as arm,
            mock.patch.object(kr.spec_decode, "release") as release,
            mock.patch.object(
                kr, "forward_pass", side_effect=RuntimeError("synthetic")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                kr._spec_step_deep(
                    [], object(), lambda ids: ids, [1, 2], 2, 1
                )
        self.assertEqual((arm.call_count, release.call_count), (1, 1))


class PackedHeadLifecycleTests(unittest.TestCase):
    def test_dense_fallback_is_not_followed_by_packed_reload(self):
        sentinel = torch.ones(2, 2)
        prior = (
            kr.INT8_LM_HEAD,
            kr._LM_Q,
            kr._LM_SC,
            kr._LM_W,
        )
        try:
            kr.INT8_LM_HEAD = True
            kr._LM_Q = None
            kr._LM_SC = None
            kr._LM_W = sentinel
            with mock.patch.object(kr, "_load_int8_packed") as load_packed:
                kr._ensure_lm_head_loaded()
            load_packed.assert_not_called()
            self.assertIs(kr._LM_W, sentinel)
            self.assertIsNone(kr._LM_Q)
        finally:
            (
                kr.INT8_LM_HEAD,
                kr._LM_Q,
                kr._LM_SC,
                kr._LM_W,
            ) = prior


if __name__ == "__main__":
    unittest.main()
