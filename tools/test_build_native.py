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

BASE_FLAGS = {
    next(iter(aliases))
    for aliases in native.X86_64_BASE_FEATURES.values()
}


class FakeFunction:
    def __init__(self, result=None, callback=None):
        self.result = result
        self.callback = callback
        self.argtypes = None
        self.restype = None
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
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
        cuda_abi: int | None = None,
        cuda_shapes: tuple[int, int, int, int] | None = None,
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
        if cuda_abi is not None:
            self.k3_cuda_moe_abi_version = FakeFunction(cuda_abi)
        if cuda_shapes is not None:
            def report_cuda_shapes(hidden, intermediate, span, pointer_layout):
                ctypes.cast(hidden, ctypes.POINTER(ctypes.c_int)).contents.value = (
                    cuda_shapes[0]
                )
                ctypes.cast(
                    intermediate, ctypes.POINTER(ctypes.c_int)
                ).contents.value = cuda_shapes[1]
                ctypes.cast(
                    span, ctypes.POINTER(ctypes.c_int64)
                ).contents.value = cuda_shapes[2]
                ctypes.cast(
                    pointer_layout, ctypes.POINTER(ctypes.c_uint32)
                ).contents.value = cuda_shapes[3]
            self.k3_cuda_moe_shapes = FakeFunction(
                callback=report_cuda_shapes
            )


class TargetTests(unittest.TestCase):
    def test_supported_target_flags(self):
        darwin = native.detect_target("darwin", "arm64")
        self.assertEqual(darwin.c_arch_flags, ("-mcpu=native",))
        self.assertTrue(darwin.supports_metal)

        x86 = native.detect_target("linux", "x86_64")
        self.assertEqual(
            x86.c_arch_flags,
            (
                "-march=x86-64",
                "-mtune=native",
                "-msse3",
                "-mssse3",
                "-mavx",
                "-mfma",
            ),
        )
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
        native.preflight_target(target, cpu_flags=BASE_FLAGS)
        # AVX2 is an optional runtime-selected island, not a load-time contract.
        native.preflight_target(target, cpu_flags=BASE_FLAGS | {"avx2"})
        with self.assertRaisesRegex(native.BuildError, "missing CPU flags: fma"):
            native.preflight_target(target, cpu_flags=BASE_FLAGS - {"fma"})

    def test_x86_artifacts_require_both_dispatch_islands(self):
        x86 = native.artifacts_for(
            native.detect_target("linux", "x86_64"), skip_metal=True
        )
        arm = native.artifacts_for(
            native.detect_target("linux", "aarch64"), skip_metal=True
        )
        for artifact in x86:
            self.assertTrue(
                set(native.X86_AVX2_SYMBOLS).issubset(
                    artifact.required_symbols
                )
            )
        for artifact in arm:
            self.assertTrue(
                set(native.X86_AVX2_SYMBOLS).isdisjoint(
                    artifact.required_symbols
                )
            )

    def test_cuda_artifact_selection_is_explicit_and_deterministic(self):
        linux = native.detect_target("linux", "x86_64")
        darwin = native.detect_target("darwin", "arm64")

        self.assertFalse(
            any(
                artifact.language == "cuda"
                for artifact in native.artifacts_for(
                    linux, cuda_mode="off", nvcc_available=True
                )
            )
        )
        self.assertFalse(
            any(
                artifact.language == "cuda"
                for artifact in native.artifacts_for(
                    linux, cuda_mode="auto", nvcc_available=False
                )
            )
        )
        auto = native.artifacts_for(
            linux, cuda_mode="auto", nvcc_available=True
        )
        required = native.artifacts_for(
            linux, cuda_mode="on", nvcc_available=False
        )
        for artifacts in (auto, required):
            cuda = [item for item in artifacts if item.language == "cuda"]
            self.assertEqual(len(cuda), 1)
            self.assertEqual(cuda[0].destination.name, "libcudamoe.so")
            self.assertEqual(cuda[0].expected_cuda_shapes, native.CUDA_SHAPES)

        self.assertFalse(
            any(
                artifact.language == "cuda"
                for artifact in native.artifacts_for(
                    darwin, cuda_mode="auto", nvcc_available=True
                )
            )
        )
        with self.assertRaisesRegex(native.BuildError, "only on Linux"):
            native.artifacts_for(darwin, cuda_mode="on")

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
        self.assertIn("-march=x86-64", command)
        self.assertIn("-mtune=native", command)
        self.assertIn("-mavx", command)
        self.assertIn("-mfma", command)
        self.assertNotIn("-mavx2", command)
        self.assertNotIn("-march=x86-64-v3", command)
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

    def test_cuda_command_uses_portable_floor_and_ptx_fallback(self):
        target = native.detect_target("linux", "x86_64")
        artifact = [
            item
            for item in native.artifacts_for(
                target, cuda_mode="on", nvcc_available=True
            )
            if item.language == "cuda"
        ][0]
        command = native.compile_command(
            artifact,
            target,
            Path("/tmp/libcudamoe.so"),
            cuda_compiler=[sys.executable, "--nvcc-wrapper"],
        )
        self.assertEqual(command[:2], [sys.executable, "--nvcc-wrapper"])
        self.assertIn("-Xcompiler=-fPIC", command)
        self.assertIn("-gencode=arch=compute_75,code=sm_75", command)
        self.assertIn("-gencode=arch=compute_75,code=compute_75", command)
        self.assertFalse(any("sm_100" in part for part in command))
        self.assertFalse(any("sm_120" in part for part in command))
        self.assertEqual(command[-2:], ["-o", "/tmp/libcudamoe.so"])

    def test_nvcc_environment_is_parsed_and_resolved_safely(self):
        requested = []

        def find(executable):
            requested.append(executable)
            return "/opt/cuda/bin/nvcc"

        argv = native.cuda_compiler_argv(
            {"NVCC": "custom-nvcc --use_fast_math"}, finder=find
        )
        self.assertEqual(requested, ["custom-nvcc"])
        self.assertEqual(
            argv, ["/opt/cuda/bin/nvcc", "--use_fast_math"]
        )
        self.assertIsNone(
            native.cuda_compiler_argv({}, finder=lambda _name: None)
        )
        with self.assertRaisesRegex(native.BuildError, "NVCC is empty"):
            native.cuda_compiler_argv({"NVCC": "   "})

        def broken_path(_name):
            raise OSError("synthetic PATH failure")

        with self.assertRaisesRegex(native.BuildError, "could not locate NVCC"):
            native.cuda_compiler_argv({}, finder=broken_path)


class ParserTests(unittest.TestCase):
    def test_cuda_mode_defaults_to_auto_and_accepts_all_modes(self):
        parser = native._parser()
        self.assertEqual(parser.parse_args([]).cuda, "auto")
        for mode in ("auto", "on", "off"):
            self.assertEqual(parser.parse_args([f"--cuda={mode}"]).cuda, mode)


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

    def test_cuda_symbols_abi_and_layout_without_device_probe(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._path(td)
            artifact = native.Artifact(
                "CUDA MoE",
                path,
                path,
                "cuda",
                native.CUDA_SYMBOLS,
                abi_version=native.CUDA_ABI_VERSION,
                abi_symbol="k3_cuda_moe_abi_version",
                expected_cuda_shapes=native.CUDA_SHAPES,
            )
            good = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=native.CUDA_SHAPES,
            )
            native.validate_artifact(
                path, artifact, cdll_factory=lambda _: good
            )
            self.assertEqual(good.k3_cuda_moe_abi_version.calls, 1)
            self.assertEqual(good.k3_cuda_moe_shapes.calls, 1)
            self.assertEqual(good.k3_cuda_moe_available.calls, 0)
            self.assertEqual(good.k3_cuda_moe_launch.calls, 0)

            bad_abi = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=99,
                cuda_shapes=native.CUDA_SHAPES,
            )
            with self.assertRaisesRegex(native.BuildError, "has ABI 99"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad_abi
                )

            bad_span = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=(3584, 3072, 1, 1),
            )
            with self.assertRaisesRegex(
                native.BuildError, "shape/layout handshake returned"
            ):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad_span
                )

            bad_layout = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=(3584, 3072, 17_547_264, 99),
            )
            with self.assertRaisesRegex(
                native.BuildError, "shape/layout handshake returned"
            ):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad_layout
                )

            missing = FakeLibrary(
                tuple(
                    symbol
                    for symbol in native.CUDA_SYMBOLS
                    if symbol != "k3_cuda_last_error"
                ),
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=native.CUDA_SHAPES,
            )
            with self.assertRaisesRegex(
                native.BuildError, "k3_cuda_last_error"
            ):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: missing
                )


class AtomicBuildTests(unittest.TestCase):
    def _tree(
        self,
        root: Path,
        *,
        suffix: str = ".so",
        cuda: bool = False,
    ) -> Path:
        tools = root / "repo" / "tools"
        tools.mkdir(parents=True)
        (tools / "fused_gemv.c").write_text("gemv", encoding="utf-8")
        (tools / "fused_gemv_batch.c").write_text("batch", encoding="utf-8")
        (tools / f"libmxfp4gemv{suffix}").write_bytes(b"old-gemv")
        (tools / f"libmxfp4batch{suffix}").write_bytes(b"old-batch")
        if cuda:
            (tools / "cuda_moe_kernels.cu").write_text(
                "cuda", encoding="utf-8"
            )
            (tools / "libcudamoe.so").write_bytes(b"old-cuda")
        return tools

    @staticmethod
    def _write_output(command, _cwd):
        output = Path(command[command.index("-o") + 1])
        source = next(
            Path(part).name for part in command
            if str(part).endswith((".c", ".mm", ".cu"))
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
                    cuda_mode="off",
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
                    cuda_mode="off",
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
                cuda_mode="off",
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

    def test_darwin_auto_keeps_outputs_and_never_probes_nvcc(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), suffix=".dylib")

            def forbidden_probe(_name):
                self.fail("Darwin auto mode must not probe NVCC")

            outputs = native.build_native(
                target=native.detect_target("darwin", "arm64"),
                tools_dir=tools,
                skip_metal=True,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=lambda _path, _artifact: None,
                nvcc_finder=forbidden_probe,
            )
            self.assertEqual(
                [path.name for path in outputs],
                ["libmxfp4gemv.dylib", "libmxfp4batch.dylib"],
            )

    def test_linux_auto_without_nvcc_builds_required_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))
            probes = []

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=lambda _path, _artifact: None,
                nvcc_finder=lambda name: probes.append(name),
            )
            self.assertEqual(probes, ["nvcc"])
            self.assertEqual(len(outputs), 2)
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )

    def test_cuda_off_never_probes_or_builds_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            commands = []

            def forbidden_probe(_name):
                self.fail("CUDA off mode must not probe NVCC")

            def record(command, cwd):
                commands.append(command)
                self._write_output(command, cwd)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="off",
                environ={"CC": sys.executable},
                runner=record,
                validator=lambda _path, _artifact: None,
                nvcc_finder=forbidden_probe,
            )
            self.assertEqual(len(outputs), 2)
            self.assertEqual(len(commands), 2)
            self.assertFalse(
                any(
                    any(str(part).endswith(".cu") for part in command)
                    for command in commands
                )
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_cuda_auto_compile_failure_does_not_block_cpu_install(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            reports = []

            def fail_cuda(command, cwd):
                self._write_output(command, cwd)
                if any(str(part).endswith(".cu") for part in command):
                    raise native.BuildError("synthetic nvcc rejection")

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=fail_cuda,
                validator=lambda _path, _artifact: None,
                nvcc_finder=lambda _name: sys.executable,
                reporter=reports.append,
            )
            self.assertEqual(len(outputs), 2)
            self.assertIn("synthetic nvcc rejection", reports[0])
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(),
                b"new:fused_gemv_batch.c",
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_cuda_auto_validation_failure_does_not_block_cpu_install(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            reports = []

            def reject_cuda(_path, artifact):
                if artifact.language == "cuda":
                    raise ValueError("synthetic third-party validator failure")

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=reject_cuda,
                nvcc_finder=lambda _name: sys.executable,
                reporter=reports.append,
            )
            self.assertEqual(len(outputs), 2)
            self.assertIn("third-party validator failure", reports[0])
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_cuda_on_failure_is_clear_and_preserves_all_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)

            def reject_cuda(_path, artifact):
                if artifact.language == "cuda":
                    raise native.BuildError("synthetic CUDA ABI rejection")

            with self.assertRaisesRegex(
                native.BuildError,
                "required by --cuda=on.*synthetic CUDA ABI rejection",
            ):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    cuda_mode="on",
                    environ={"CC": sys.executable},
                    runner=self._write_output,
                    validator=reject_cuda,
                    nvcc_finder=lambda _name: sys.executable,
                )
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )
            self.assertFalse(list(tools.glob(".*.build-*")))

    def test_cuda_on_success_installs_after_every_validation(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            validated = []

            def accept(path, artifact):
                self.assertTrue(path.read_bytes().startswith(b"new:"))
                self.assertEqual(
                    (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
                )
                self.assertEqual(
                    (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
                )
                self.assertEqual(
                    (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
                )
                validated.append(artifact.label)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="on",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=accept,
                nvcc_finder=lambda _name: sys.executable,
            )
            self.assertEqual(
                validated, ["MXFP4 GEMV", "MXFP4 batch", "CUDA MoE"]
            )
            self.assertEqual(len(outputs), 3)
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(),
                b"new:cuda_moe_kernels.cu",
            )

    def test_cuda_auto_install_failure_does_not_roll_back_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            reports = []

            def reject_cuda_replace(source, destination):
                if Path(destination).name == "libcudamoe.so":
                    raise OSError("synthetic CUDA install rejection")
                os.replace(source, destination)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=lambda _path, _artifact: None,
                nvcc_finder=lambda _name: sys.executable,
                reporter=reports.append,
                replace=reject_cuda_replace,
            )
            self.assertEqual(len(outputs), 2)
            self.assertIn("synthetic CUDA install rejection", reports[0])
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(),
                b"new:fused_gemv_batch.c",
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_cuda_on_missing_nvcc_fails_before_required_build(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            commands = []

            with self.assertRaisesRegex(native.BuildError, "NVCC was not found"):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    cuda_mode="on",
                    environ={"CC": sys.executable},
                    runner=lambda command, _cwd: commands.append(command),
                    validator=lambda _path, _artifact: None,
                    nvcc_finder=lambda _name: None,
                )
            self.assertEqual(commands, [])

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
