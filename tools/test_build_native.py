#!/usr/bin/env python3
"""Weight-free unit tests for tools/build_native.py."""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_native as native

V3_FLAGS = {
    next(iter(aliases))
    for aliases in native.X86_64_V3_FEATURES.values()
}


class FakeFunction:
    def __init__(self, result=None, callback=None):
        self.result = result
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        if self.callback is not None:
            return self.callback(*args)
        return self.result


class FakeLibrary:
    def __init__(
        self,
        symbols: tuple[str, ...],
        *,
        abi: int | None = None,
        shapes: tuple[int, int, int] | None = None,
    ):
        for symbol in symbols:
            setattr(self, symbol, FakeFunction())
        if abi is not None:
            self.mxfp4_abi_version = FakeFunction(abi)
        if shapes is not None:
            def report_shapes(hidden, intermediate, span):
                ctypes.cast(hidden, ctypes.POINTER(ctypes.c_int)).contents.value = shapes[0]
                ctypes.cast(
                    intermediate, ctypes.POINTER(ctypes.c_int)
                ).contents.value = shapes[1]
                ctypes.cast(
                    span, ctypes.POINTER(ctypes.c_longlong)
                ).contents.value = shapes[2]
            self.k3_metal_shapes = FakeFunction(callback=report_shapes)


class TargetTests(unittest.TestCase):
    def test_supported_target_flags(self):
        darwin = native.detect_target("darwin", "arm64")
        self.assertEqual(darwin.c_arch_flags, ("-mcpu=native",))
        self.assertTrue(darwin.supports_metal)

        x86 = native.detect_target("linux", "x86_64")
        self.assertEqual(x86.c_arch_flags, ("-march=x86-64-v3",))
        self.assertEqual(x86.suffix, ".so")

        arm = native.detect_target("linux", "aarch64")
        self.assertEqual(arm.c_arch_flags, ("-mcpu=native",))
        self.assertFalse(arm.supports_metal)

    def test_unsupported_targets_are_clear(self):
        with self.assertRaisesRegex(native.BuildError, "Apple Silicon"):
            native.detect_target("darwin", "x86_64")
        with self.assertRaisesRegex(native.BuildError, "supported architectures"):
            native.detect_target("linux", "riscv64")
        with self.assertRaisesRegex(native.BuildError, "unsupported platform"):
            native.detect_target("win32", "amd64")

    def test_x86_cpu_preflight(self):
        target = native.detect_target("linux", "x86_64")
        native.preflight_target(target, cpu_flags=V3_FLAGS)
        with self.assertRaisesRegex(native.BuildError, "missing CPU flags: avx2"):
            native.preflight_target(target, cpu_flags=V3_FLAGS - {"avx2"})

    def test_cpuinfo_parser(self):
        with tempfile.TemporaryDirectory() as td:
            cpuinfo = Path(td) / "cpuinfo"
            cpuinfo.write_text(
                "processor : 0\nflags : sse2 ssse3 fma avx2\n"
                "processor : 1\nflags : sse2 ssse3 fma avx2 bmi2\n",
                encoding="utf-8",
            )
            flags = native.read_linux_cpu_flags(cpuinfo)
        self.assertEqual(flags, {"sse2", "ssse3", "fma", "avx2"})


class CommandTests(unittest.TestCase):
    def test_linux_commands_honor_cc_and_architecture(self):
        target = native.detect_target("linux", "x86_64")
        artifact = native.Artifact(
            "gemv",
            Path("/repo/tools/fused_gemv.c"),
            Path("/repo/tools/libmxfp4gemv.so"),
            "c",
            native.GEMV_SYMBOLS,
            abi_version=1,
        )
        command = native.compile_command(
            artifact,
            target,
            Path("/tmp/out.so"),
            environ={"CC": f"{sys.executable} --compiler-wrapper"},
        )
        self.assertEqual(command[:2], [sys.executable, "--compiler-wrapper"])
        self.assertIn("-march=x86-64-v3", command)
        self.assertIn("-fPIC", command)
        self.assertIn("-shared", command)
        self.assertNotIn("-dynamiclib", command)

    def test_darwin_metal_command_honors_cxx(self):
        target = native.detect_target("darwin", "arm64")
        artifact = native.artifacts_for(target)[2]
        command = native.compile_command(
            artifact,
            target,
            Path("/tmp/metal.dylib"),
            environ={"CXX": f"{sys.executable} --cxx-wrapper"},
        )
        self.assertEqual(command[:2], [sys.executable, "--cxx-wrapper"])
        self.assertIn("-fobjc-arc", command)
        self.assertIn("-framework", command)
        self.assertIn("Metal", command)

    def test_missing_compiler_is_clear(self):
        target = native.detect_target("linux", "aarch64")
        artifact = native.Artifact(
            "gemv",
            Path("/repo/fused_gemv.c"),
            Path("/repo/libmxfp4gemv.so"),
            "c",
            native.GEMV_SYMBOLS,
            abi_version=1,
        )
        with self.assertRaisesRegex(native.BuildError, "CC compiler not found"):
            native.compile_command(
                artifact,
                target,
                Path("/tmp/out.so"),
                environ={"CC": "definitely-not-a-real-deltafin-compiler"},
            )


class ValidationTests(unittest.TestCase):
    def _path(self, directory: str) -> Path:
        path = Path(directory) / "candidate.so"
        path.write_bytes(b"candidate")
        return path

    def test_abi_and_symbols(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._path(td)
            artifact = native.Artifact(
                "gemv", path, path, "c", native.GEMV_SYMBOLS, abi_version=1
            )
            library = FakeLibrary(native.GEMV_SYMBOLS, abi=1)
            native.validate_artifact(
                path, artifact, cdll_factory=lambda _: library
            )

            bad = FakeLibrary(native.GEMV_SYMBOLS, abi=9)
            with self.assertRaisesRegex(native.BuildError, "has ABI 9"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad
                )

            missing = FakeLibrary(("mxfp4_gemv",), abi=1)
            with self.assertRaisesRegex(native.BuildError, "missing required symbols"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: missing
                )

    def test_metal_symbol_and_shape_handshake(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._path(td)
            artifact = native.Artifact(
                "Metal",
                path,
                path,
                "cxx",
                native.METAL_SYMBOLS,
                expected_metal_shapes=native.METAL_SHAPES,
            )
            good = FakeLibrary(
                native.METAL_SYMBOLS, shapes=native.METAL_SHAPES
            )
            native.validate_artifact(
                path, artifact, cdll_factory=lambda _: good
            )
            bad = FakeLibrary(
                native.METAL_SYMBOLS, shapes=(3584, 3072, 1)
            )
            with self.assertRaisesRegex(native.BuildError, "shape handshake returned"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad
                )


class AtomicBuildTests(unittest.TestCase):
    def _tree(self, root: Path) -> Path:
        tools = root / "repo" / "tools"
        tools.mkdir(parents=True)
        (tools / "fused_gemv.c").write_text("gemv", encoding="utf-8")
        (tools / "fused_gemv_batch.c").write_text("batch", encoding="utf-8")
        (tools / "libmxfp4gemv.so").write_bytes(b"old-gemv")
        (tools / "libmxfp4batch.so").write_bytes(b"old-batch")
        return tools

    @staticmethod
    def _write_output(command, _cwd):
        output = Path(command[command.index("-o") + 1])
        source = next(
            Path(part).name for part in command
            if str(part).endswith((".c", ".mm"))
        )
        output.write_bytes(f"new:{source}".encode())

    def test_compile_failure_preserves_every_existing_library(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))
            calls = 0

            def fail_second(command, cwd):
                nonlocal calls
                calls += 1
                self._write_output(command, cwd)
                if calls == 2:
                    raise native.BuildError("synthetic compiler failure")

            with self.assertRaisesRegex(native.BuildError, "synthetic compiler"):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    environ={"CC": sys.executable},
                    runner=fail_second,
                    validator=lambda _path, _artifact: None,
                )
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
            )

    def test_validation_failure_preserves_every_existing_library(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))

            def reject_batch(_path, artifact):
                if artifact.label == "MXFP4 batch":
                    raise native.BuildError("synthetic ABI rejection")

            with self.assertRaisesRegex(native.BuildError, "synthetic ABI"):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    environ={"CC": sys.executable},
                    runner=self._write_output,
                    validator=reject_batch,
                )
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
            )
            self.assertFalse(list(tools.glob(".*.build-*")))

    def test_success_installs_only_after_all_validations(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))
            validated = []

            def accept(path, artifact):
                self.assertTrue(path.read_bytes().startswith(b"new:"))
                validated.append(artifact.label)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=accept,
            )
            self.assertEqual(
                validated, ["MXFP4 GEMV", "MXFP4 batch"]
            )
            self.assertEqual(len(outputs), 2)
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(),
                b"new:fused_gemv_batch.c",
            )

    def test_install_error_rolls_back_prior_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            destination_a = root / "a.so"
            destination_b = root / "b.so"
            temporary_a = root / "new-a.so"
            temporary_b = root / "new-b.so"
            destination_a.write_bytes(b"old-a")
            destination_b.write_bytes(b"old-b")
            temporary_a.write_bytes(b"new-a")
            temporary_b.write_bytes(b"new-b")
            artifact_a = native.Artifact(
                "a", root / "a.c", destination_a, "c", ()
            )
            artifact_b = native.Artifact(
                "b", root / "b.c", destination_b, "c", ()
            )
            calls = 0

            def fail_second(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic replace failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(native.BuildError, "synthetic replace"):
                native.install_validated(
                    [(artifact_a, temporary_a), (artifact_b, temporary_b)],
                    replace=fail_second,
                )
            self.assertEqual(destination_a.read_bytes(), b"old-a")
            self.assertEqual(destination_b.read_bytes(), b"old-b")
            self.assertFalse(list(root.glob(".*.backup-*")))


if __name__ == "__main__":
    unittest.main()
