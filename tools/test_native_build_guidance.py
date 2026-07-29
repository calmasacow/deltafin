#!/usr/bin/env python3
"""Weight-free regression tests for actionable missing-library guidance."""

from __future__ import annotations

import os
import pathlib
import shlex
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runtime_platform as rp
import metal_moe


class NativeBuildGuidanceTests(unittest.TestCase):
    def test_command_uses_active_python_and_quotes_checkout_path(self):
        tools = os.path.join(os.sep, "tmp", "Deltafin checkout", "tools")
        command = rp.native_build_command(
            tools, executable="/tmp/venv with spaces/bin/python"
        )
        self.assertEqual(
            shlex.split(command),
            [
                "/tmp/venv with spaces/bin/python",
                os.path.join(tools, "build_native.py"),
            ],
        )

    def test_missing_native_library_names_one_complete_rebuild_command(self):
        with tempfile.TemporaryDirectory(prefix="Deltafin checkout ") as td:
            tools = os.path.join(td, "tools")
            os.mkdir(tools)
            with mock.patch.object(sys, "executable", "/tmp/venv/bin/python"):
                with self.assertRaises(rp.NativeLibraryError) as raised:
                    rp.load_native_library(
                        tools,
                        "libmxfp4batch",
                        env_var="K3_BATCH_LIB",
                        required_symbols=("mxfp4_moe_layer",),
                        platform="darwin",
                        machine="arm64",
                    )
            message = str(raised.exception)
            self.assertIn("required native library not found", message)
            self.assertIn("build_native.py", message)
            self.assertIn("/tmp/venv/bin/python", message)
            self.assertIn("K3_BATCH_LIB", message)
            command = message.split("with: ", 1)[1].split(" ; or ", 1)[0]
            self.assertEqual(
                shlex.split(command),
                [
                    "/tmp/venv/bin/python",
                    os.path.join(tools, "build_native.py"),
                ],
            )

    def test_missing_metal_library_includes_the_same_rebuild_path(self):
        with mock.patch.object(metal_moe, "_lib", None):
            with mock.patch.object(metal_moe, "_load_error", None):
                with mock.patch.object(
                    metal_moe.ctypes,
                    "CDLL",
                    side_effect=OSError("synthetic missing library"),
                ):
                    with mock.patch.object(
                        sys, "executable", "/tmp/venv/bin/python"
                    ):
                        self.assertIsNone(metal_moe._load())
                        message = metal_moe.last_error()
        self.assertIn("synthetic missing library", message)
        self.assertIn("/tmp/venv/bin/python", message)
        self.assertIn(
            os.path.join(str(HERE), "build_native.py"),
            message,
        )


if __name__ == "__main__":
    unittest.main()
