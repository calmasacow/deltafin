#!/usr/bin/env python3
"""Exact/lifetime tests for the local embedding-row handoff."""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr


def _bits(tensor):
    return tensor.squeeze(0).contiguous().view(torch.uint16).numpy().tobytes()


class LazyEmbedDirectReadTests(unittest.TestCase):
    def setUp(self):
        self.rowbytes = kr.H * 2
        self.rows = [
            bytes(((row * 37 + offset * 13) & 0xFF)
                  for offset in range(self.rowbytes))
            for row in range(4)
        ]
        self.temp = tempfile.TemporaryDirectory(prefix="deltafin-embed-test-")
        self.path = Path(self.temp.name) / "embed.bin"
        self.path.write_bytes(b"".join(self.rows))
        self.embed = kr.LazyEmbed.__new__(kr.LazyEmbed)
        self.embed.path = str(self.path)
        self.embed.meta = {}
        self.embed.rowbytes = self.rowbytes
        self.embed._fd = None
        self.embed._ensure_fd()

    def tearDown(self):
        self.embed.close()
        self.temp.cleanup()

    def _call(self, ids):
        with (
            mock.patch.object(kr, "DEV", torch.device("cpu")),
            mock.patch.object(kr, "DT", torch.bfloat16),
        ):
            return self.embed(ids)

    @unittest.skipUnless(hasattr(os, "preadv"), "preadv unavailable")
    def test_t1_preadv_fills_final_owner_with_exact_bits(self):
        with mock.patch.object(kr, "_PREADV", wraps=os.preadv) as preadv:
            result = self._call([2])
        self.assertEqual(_bits(result), self.rows[2])
        self.assertEqual(preadv.call_count, 1)

        # The tensor must retain the bytearray owner when CPU .to() aliases it.
        self.embed.close()
        gc.collect()
        junk = [bytearray(self.rowbytes) for _ in range(8)]
        self.assertEqual(_bits(result), self.rows[2])
        self.assertEqual(len(junk), 8)

    @unittest.skipUnless(hasattr(os, "preadv"), "preadv unavailable")
    def test_positive_partial_preadv_is_completed_exactly(self):
        calls = 0

        def partial(fd, buffers, offset):
            nonlocal calls
            calls += 1
            target = buffers[0]
            count = min(len(target), max(1, self.rowbytes // 5))
            chunk = os.pread(fd, count, offset)
            target[:len(chunk)] = chunk
            return len(chunk)

        with mock.patch.object(kr, "_PREADV", partial):
            owner = self.embed._local_row_buffer(1)
        self.assertEqual(bytes(owner), self.rows[1])
        self.assertGreater(calls, 2)

    @unittest.skipUnless(hasattr(os, "preadv"), "preadv unavailable")
    def test_preadv_propagates_original_os_error(self):
        for error in (
            InterruptedError("synthetic signal-handler error"),
            OSError("synthetic storage error"),
        ):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(
                    kr, "_PREADV", side_effect=error
                ):
                    with self.assertRaises(type(error)) as caught:
                        self.embed._local_row_buffer(1)
                self.assertIs(caught.exception, error)

    @unittest.skipUnless(hasattr(os, "preadv"), "preadv unavailable")
    def test_short_preadv_retains_fail_closed_contract(self):
        with self.assertRaisesRegex(
            IOError, rf"short embedding read 0/{self.rowbytes}"
        ):
            self.embed._local_row_buffer(99)

    def test_missing_preadv_uses_existing_pread_path(self):
        with mock.patch.object(kr, "_PREADV", None):
            result = self._call([3])
        self.assertEqual(_bits(result), self.rows[3])

    def test_multirow_order_and_duplicates_keep_coalesced_path(self):
        ids = [2, 1, 2, 3]
        with mock.patch.object(
            self.embed,
            "_local_row_buffer",
            side_effect=AssertionError("T>1 must keep the coalesced path"),
        ):
            result = self._call(ids)
        self.assertEqual(_bits(result), b"".join(self.rows[i] for i in ids))

    def test_remote_fallback_never_uses_local_preadv(self):
        self.embed.close()
        self.embed.path = str(Path(self.temp.name) / "missing.bin")
        with (
            mock.patch.object(self.embed, "_row",
                              side_effect=[self.rows[1], self.rows[3]]) as row,
            mock.patch.object(kr, "_PREADV") as preadv,
        ):
            result = self._call([1, 3])
        self.assertEqual(_bits(result), self.rows[1] + self.rows[3])
        self.assertEqual(row.call_count, 2)
        preadv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
