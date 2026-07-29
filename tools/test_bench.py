#!/usr/bin/env python3
"""Lightweight tests for tools/bench.py; no model weights are loaded."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402


FAKE_RUNNER = r"""
import argparse
import json
import os
import sys
import time

SCHEMA = "deltafin.run_event.v1"
ap = argparse.ArgumentParser()
ap.add_argument("--prompt")
ap.add_argument("--max-new", type=int)
ap.add_argument("--events-jsonl", required=True)
ap.add_argument("--chat", action="store_true")
args = ap.parse_args()

def event(kind, **fields):
    record = {
        "schema": SCHEMA,
        "event": kind,
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.perf_counter_ns(),
        **fields,
    }
    with open(args.events_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()

event("run_start", input_token_ids=[7, 8], config={"fake": True})
if os.environ.get("K3_FAKE_FAIL") == "1":
    print("intentional failure", file=sys.stderr)
    raise SystemExit(7)
event("prefill_done", duration_ns=1_250_000_000, emitted_token_ids=[10])
event("decode_step", step=1, duration_ns=2_000_000_000,
      emitted_token_ids=[11])
event("decode_step", step=2, duration_ns=500_000_000,
      emitted_token_ids=[12, 13])
event("run_end", status="ok", duration_ns=3_750_000_000,
      emitted_token_ids=[10, 11, 12, 13],
      completion_token_ids=[10, 11, 12, 13],
      completion_text="fake completion")
print("completion: fake completion")
print("token ids: [10, 11, 12, 13]")
"""


class BenchUnitTests(unittest.TestCase):
    def test_env_delta_is_shell_like_but_not_executed(self):
        self.assertEqual(
            bench.parse_env_delta('K3_A=1 K3_LABEL="two words" K3_EMPTY='),
            {"K3_A": "1", "K3_LABEL": "two words", "K3_EMPTY": ""},
        )
        with self.assertRaises(ValueError):
            bench.parse_env_delta("K3_OK=1 bare-word")

    def test_structured_metrics_count_emitted_tokens(self):
        events = [
            {
                "schema": bench.EVENT_SCHEMA,
                "event": "run_start",
                "input_token_ids": [1, 2],
                "config": {"device": "fake"},
            },
            {
                "schema": bench.EVENT_SCHEMA,
                "event": "prefill_done",
                "duration_ns": 1_250_000_000,
                "emitted_token_ids": [10],
            },
            {
                "schema": bench.EVENT_SCHEMA,
                "event": "decode_step",
                "step": 1,
                "duration_ns": 2_000_000_000,
                "emitted_token_ids": [11],
            },
            {
                "schema": bench.EVENT_SCHEMA,
                "event": "decode_step",
                "step": 2,
                "duration_ns": 500_000_000,
                "emitted_token_ids": [12, 13],
            },
            {
                "schema": bench.EVENT_SCHEMA,
                "event": "run_end",
                "status": "ok",
                "duration_ns": 3_750_000_000,
                "emitted_token_ids": [10, 11, 12, 13],
                "completion_token_ids": [10, 11, 12, 13],
                "completion_text": "fake completion",
                "runtime": {
                    "int8_kda_qkv": {
                        "controllers_installed": 2,
                        "enabled_at_end": True,
                        "packed_project_calls": 6,
                    },
                },
            },
        ]
        parsed = bench.parse_structured_events(events, warmup_steps=1)
        self.assertEqual(parsed["parse_errors"], [])
        self.assertEqual(parsed["prefill_s"], 1.25)
        self.assertEqual(parsed["steady_decode_tokens"], 2)
        self.assertEqual(parsed["steady_decode_ns"], 500_000_000)
        self.assertEqual(parsed["steady_tps"], 4.0)
        self.assertEqual(parsed["completion_token_ids"], [10, 11, 12, 13])
        self.assertEqual(
            parsed["runner_runtime"]["int8_kda_qkv"]["packed_project_calls"],
            6,
        )


class BenchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.runner = self.root / "fake_runner.py"
        self.runner.write_text(textwrap.dedent(FAKE_RUNNER), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_bench(self, *extra):
        output = self.root / ("evidence-" + str(len(list(self.root.glob("evidence-*")))))
        argv = [
            "--runner",
            str(self.runner),
            "--python",
            sys.executable,
            "--output-dir",
            str(output),
            "--tokens",
            "4",
            *extra,
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.main(argv), output

    def test_campaign_persists_raw_and_aggregate_evidence(self):
        code, output = self.run_bench(
            "--configs",
            "",
            "K3_FAKE_VARIANT=same",
            "--names",
            "reference",
            "candidate",
            "--reps",
            "2",
        )
        self.assertEqual(code, 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertTrue(summary["all_runs_valid"])
        self.assertTrue(summary["all_outputs_exact"])
        self.assertEqual(summary["attempted_runs"], 4)
        self.assertEqual(
            summary["exact_oracle"]["completion_token_ids"], [10, 11, 12, 13]
        )
        self.assertEqual(
            summary["configs"]["candidate"]["metrics"]["steady_tps"]["median"],
            4.0,
        )
        self.assertEqual(len((output / "runs.jsonl").read_text().splitlines()), 4)
        for run_number in range(1, 5):
            matches = list(output.glob(f"run-{run_number:03d}-*"))
            self.assertEqual(len(matches), 1)
            self.assertTrue((matches[0] / "stdout.log").exists())
            self.assertTrue((matches[0] / "stderr.log").exists())
            self.assertTrue((matches[0] / "events.jsonl").exists())
            self.assertTrue((matches[0] / "result.json").exists())

    def test_nonzero_runner_is_invalid_and_nonzero(self):
        code, output = self.run_bench(
            "--configs", "K3_FAKE_FAIL=1", "--names", "broken", "--reps", "2"
        )
        self.assertEqual(code, 2)
        summary = json.loads((output / "summary.json").read_text())
        self.assertFalse(summary["all_runs_valid"])
        self.assertTrue(summary["stopped_early"])
        self.assertEqual(summary["attempted_runs"], 1)
        result = json.loads(
            next(output.glob("run-001-*")).joinpath("result.json").read_text()
        )
        self.assertEqual(result["returncode"], 7)
        self.assertIn("runner exited with status 7", result["errors"])
        self.assertIn("intentional failure", result["stderr_tail"])


if __name__ == "__main__":
    unittest.main()
