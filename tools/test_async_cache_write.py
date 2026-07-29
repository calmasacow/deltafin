#!/usr/bin/env python3
"""Disk-light, network-free regressions for Q13 asynchronous cache writes."""
import glob
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from cache_writer import AsyncCacheWriter, CacheWriteFailure, atomic_publish


class AsyncCacheWriterTests(unittest.TestCase):
    def test_atomic_publication_keeps_old_name_visible_until_replace(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "expert.bin")
            old = b"old-complete-value"
            new = b"new-complete-value" * 1024
            with open(path, "wb") as f:
                f.write(old)
            staged = threading.Event()
            release = threading.Event()

            def before_replace(tmp, final):
                self.assertEqual(final, path)
                with open(tmp, "rb") as f:
                    self.assertEqual(f.read(), new)
                staged.set()
                self.assertTrue(release.wait(2.0))

            thread = threading.Thread(
                target=atomic_publish,
                args=(path, new),
                kwargs={"before_replace": before_replace},
            )
            thread.start()
            self.assertTrue(staged.wait(2.0))
            with open(path, "rb") as f:
                self.assertEqual(f.read(), old)
            release.set()
            thread.join(2.0)
            self.assertFalse(thread.is_alive())
            with open(path, "rb") as f:
                self.assertEqual(f.read(), new)
            self.assertEqual(glob.glob(path + ".tmp*"), [])

    def test_writer_owns_mutable_input_and_callback_runs_after_publish(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "owned.bin")
            entered = threading.Event()
            release = threading.Event()
            observed = []

            def gated_publish(final, payload):
                entered.set()
                self.assertTrue(release.wait(2.0))
                atomic_publish(final, payload)

            def on_published(final, size):
                with open(final, "rb") as f:
                    observed.append((f.read(), size))

            writer = AsyncCacheWriter(
                max_pending=1, publisher=gated_publish, name="test-owned"
            )
            source = bytearray(b"immutable snapshot")
            expected = bytes(source)
            self.assertTrue(
                writer.submit(path, source, on_published=on_published)
            )
            self.assertTrue(entered.wait(2.0))
            source[:] = b"x" * len(source)
            release.set()
            writer.shutdown()
            with open(path, "rb") as f:
                self.assertEqual(f.read(), expected)
            self.assertEqual(observed, [(expected, len(expected))])

    def test_pending_bound_backpressures_third_producer(self):
        with tempfile.TemporaryDirectory() as td:
            release = threading.Event()
            entered = threading.Event()

            def gated_publish(path, payload):
                entered.set()
                self.assertTrue(release.wait(2.0))
                atomic_publish(path, payload)

            writer = AsyncCacheWriter(
                max_pending=2, publisher=gated_publish, name="test-bound"
            )
            for i in range(2):
                self.assertTrue(
                    writer.submit(os.path.join(td, f"{i}.bin"), bytes([i]))
                )
            self.assertTrue(entered.wait(2.0))
            third_done = threading.Event()

            def submit_third():
                writer.submit(os.path.join(td, "2.bin"), b"2")
                third_done.set()

            producer = threading.Thread(target=submit_third)
            producer.start()
            self.assertFalse(third_done.wait(0.05))
            self.assertEqual(writer.snapshot()["pending"], 2)
            release.set()
            self.assertTrue(third_done.wait(2.0))
            producer.join(2.0)
            snapshot = writer.shutdown()
            self.assertEqual(snapshot["pending_peak"], 2)
            self.assertGreaterEqual(snapshot["backpressure_events"], 1)
            self.assertGreater(snapshot["backpressure_s"], 0.0)

    def test_failure_is_observable_old_file_survives_and_temp_is_cleaned(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "failure.bin")
            with open(path, "wb") as f:
                f.write(b"old")
            errors = []

            def fail_before_replace(tmp, final):
                raise OSError("injected publication failure")

            def broken_publish(final, payload):
                atomic_publish(
                    final, payload, before_replace=fail_before_replace
                )

            def on_error(final, stage, exc):
                errors.append((final, stage, type(exc).__name__))

            writer = AsyncCacheWriter(
                max_pending=1, publisher=broken_publish, name="test-failure"
            )
            writer.submit(path, b"new", on_error=on_error)
            with self.assertRaises(CacheWriteFailure):
                writer.flush()
            snapshot = writer.shutdown(raise_errors=False)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"old")
            self.assertEqual(glob.glob(path + ".tmp*"), [])
            self.assertEqual(snapshot["publish_failures"], 1)
            self.assertEqual(snapshot["failures"], 1)
            self.assertEqual(errors, [(path, "publish", "OSError")])

    def test_shutdown_drains_and_rejected_submit_supports_sync_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            writer = AsyncCacheWriter(
                max_pending=3, workers=2, name="test-shutdown"
            )
            paths = [os.path.join(td, f"{i}.bin") for i in range(3)]
            for i, path in enumerate(paths):
                self.assertTrue(writer.submit(path, bytes([i]) * 128))
            snapshot = writer.shutdown()
            self.assertEqual(snapshot["completed"], 3)
            self.assertEqual(snapshot["alive_workers"], 0)
            fallback = os.path.join(td, "fallback.bin")
            self.assertFalse(writer.submit(fallback, b"sync"))
            atomic_publish(fallback, b"sync")
            with open(fallback, "rb") as f:
                self.assertEqual(f.read(), b"sync")

    def test_shutdown_waits_for_an_admitted_backpressured_submitter(self):
        with tempfile.TemporaryDirectory() as td:
            release = threading.Event()
            first_started = threading.Event()

            def gated_publish(path, payload):
                first_started.set()
                self.assertTrue(release.wait(2.0))
                atomic_publish(path, payload)

            writer = AsyncCacheWriter(
                max_pending=1, publisher=gated_publish, name="test-race"
            )
            first = os.path.join(td, "first.bin")
            second = os.path.join(td, "second.bin")
            writer.submit(first, b"first")
            self.assertTrue(first_started.wait(2.0))
            producer = threading.Thread(
                target=lambda: writer.submit(second, b"second")
            )
            producer.start()
            deadline = time.monotonic() + 2.0
            while writer.snapshot()["submitters"] != 1:
                if time.monotonic() >= deadline:
                    self.fail("backpressured submitter was not admitted")
                time.sleep(0.001)
            result = []
            closer = threading.Thread(
                target=lambda: result.append(writer.shutdown())
            )
            closer.start()
            self.assertTrue(closer.is_alive())
            release.set()
            producer.join(2.0)
            closer.join(2.0)
            self.assertFalse(producer.is_alive())
            self.assertFalse(closer.is_alive())
            self.assertEqual(result[0]["completed"], 2)
            with open(first, "rb") as f:
                self.assertEqual(f.read(), b"first")
            with open(second, "rb") as f:
                self.assertEqual(f.read(), b"second")

    def test_fetch_v2_opt_in_and_default_sync_paths(self):
        tools = os.path.dirname(os.path.abspath(__file__))
        snippet = r"""
import json, os
import fetch_v2
payload = b"tiny-fake-expert"
path = fetch_v2._cache_path_bin(7, 11)
fetch_v2._cache_store_raw(7, 11, payload)
immediate = os.path.exists(path)
status = fetch_v2.flush_cache_writes()
assert open(path, "rb").read() == payload
print(json.dumps({
    "async": fetch_v2.ASYNC_CACHE_WRITE,
    "immediate": immediate,
    "stats": fetch_v2.stats,
    "status": status,
}))
"""
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as td:
                env = dict(os.environ)
                env["DELTAFIN_ROOT"] = td
                env["PYTHONPATH"] = tools
                env["K3_ASYNC_CACHE_WRITE"] = "1" if enabled else "0"
                env["K3_CACHE_WRITE_QUEUE"] = "2"
                proc = subprocess.run(
                    [sys.executable, "-c", snippet],
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                result = json.loads(proc.stdout.strip().splitlines()[-1])
                self.assertEqual(result["async"], enabled)
                if enabled:
                    self.assertEqual(result["stats"]["cache_write_enqueued"], 1)
                    self.assertEqual(result["status"]["completed"], 1)
                else:
                    self.assertTrue(result["immediate"])
                    self.assertEqual(result["stats"]["cache_write_sync"], 1)
                    self.assertEqual(result["status"]["completed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
