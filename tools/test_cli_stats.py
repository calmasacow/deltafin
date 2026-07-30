#!/usr/bin/env python3
"""Tests for the optional command-line live statistics display."""
from __future__ import annotations

import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr  # noqa: E402


class LiveDecodeStatsTests(unittest.TestCase):
    def test_cli_flag_is_optional(self):
        parser = kr._build_cli_parser()
        self.assertFalse(parser.parse_args([]).stats)
        self.assertTrue(parser.parse_args(["--stats"]).stats)

    def test_disabled_display_has_no_accounting_side_effect(self):
        stats = kr.LiveDecodeStats(False)
        self.assertIsNone(stats.record_prefill(2_000_000_000, 5))
        self.assertIsNone(stats.record_decode(4_000_000_000, 2))
        self.assertIsNone(stats.final_line(6_000_000_000))
        self.assertEqual(stats.decode_tokens, 0)
        self.assertEqual(stats.decode_ns, 0)

    def test_live_lines_include_running_speed_and_draft_acceptance(self):
        stats = kr.LiveDecodeStats(True)
        prefill = stats.record_prefill(2_000_000_000, 5)
        self.assertIn("prefill 2.000s", prefill)
        self.assertIn("5 prompt tokens", prefill)

        first = stats.record_decode(
            4_000_000_000,
            2,
            {"accepted_drafts": 2, "target_drafts": 2},
        )
        self.assertIn("decode 2 tok / 4.000s", first)
        self.assertIn("0.5000 tok/s", first)
        self.assertIn("2.000 s/token", first)
        self.assertIn("drafts 2/2 (100%)", first)

        second = stats.record_decode(
            6_000_000_000,
            3,
            {"accepted_drafts": 4, "target_drafts": 5},
        )
        self.assertIn("decode 5 tok / 10.000s", second)
        self.assertIn("0.5000 tok/s", second)
        self.assertIn("2.000 s/token", second)
        self.assertIn("last +3 tok in 6.000s", second)
        self.assertIn("drafts 4/5 (80%)", second)

        final = stats.final_line(12_000_000_000)
        self.assertIn("steady decode 5 tok / 10.000s", final)
        self.assertIn("model total 12.000s", final)

    def test_zero_decode_time_and_tokens_are_safe(self):
        stats = kr.LiveDecodeStats(True)
        line = stats.record_decode(0, 0)
        self.assertIn("0.0000 tok/s", line)
        self.assertIn("0.000 s/token", line)
        final = stats.final_line(0)
        self.assertIn("steady decode 0 tok / 0.000s", final)


if __name__ == "__main__":
    unittest.main()
