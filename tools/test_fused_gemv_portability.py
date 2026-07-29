#!/usr/bin/env python3
"""Synthetic, weight-free correctness gate for the native MXFP4 kernels.

The test builds the current local C sources into a temporary directory, then checks:

* every E2M1 code through the SIMD table-expansion path;
* single-thread, multi-thread, and over-limit thread requests bit-for-bit;
* the persistent batch pool against the normal exported GEMV;
* the fixed-size expert-triple launcher above its supported thread limit; and
* the ABI version exported by both native libraries.

It needs no model download or expert cache:

    python tools/test_fused_gemv_portability.py
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mxfp4 import dequant_mxfp4


U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)


def _build(source: str, output: Path) -> None:
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64", "x86_64", "amd64"}:
        raise RuntimeError(f"unsupported test architecture: {machine}")

    cc = os.environ.get("CC", "cc")
    cmd = [
        cc,
        "-O3",
        "-std=gnu11",
        "-DNO_MAIN",
        "-shared",
        str(HERE / source),
        "-o",
        str(output),
        "-lpthread",
        "-lm",
    ]
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        cmd[1:1] = ["-mcpu=native"]
    else:
        cmd[1:1] = ["-march=native", "-fPIC"]
    subprocess.run(cmd, check=True, cwd=HERE)


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.mxfp4_abi_version.argtypes = []
    lib.mxfp4_abi_version.restype = ctypes.c_uint32
    if lib.mxfp4_abi_version() != 1:
        raise AssertionError(f"{path.name}: expected ABI 1")
    return lib


def _bind_gemv(lib: ctypes.CDLL) -> None:
    lib.mxfp4_gemv.argtypes = [U8P, U8P, F32P, F32P, ctypes.c_int, ctypes.c_int]
    lib.mxfp4_gemv.restype = None
    lib.mxfp4_gemv_mt.argtypes = [
        U8P,
        U8P,
        F32P,
        F32P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mxfp4_gemv_mt.restype = None


def _gemv(lib: ctypes.CDLL, packed: np.ndarray, scales: np.ndarray, x: np.ndarray,
          nthreads: int | None = None) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.uint8)
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = packed.shape[0], packed.shape[1] * 2
    assert cols % 32 == 0
    assert scales.shape == (rows, cols // 32)
    assert x.shape == (cols,)
    out = np.full(rows, np.nan, dtype=np.float32)
    args = (
        packed.ctypes.data_as(U8P),
        scales.ctypes.data_as(U8P),
        x.ctypes.data_as(F32P),
        out.ctypes.data_as(F32P),
        rows,
        cols,
    )
    if nthreads is None:
        lib.mxfp4_gemv(*args)
    else:
        lib.mxfp4_gemv_mt(*args, nthreads)
    if np.isnan(out).any():
        raise AssertionError("GEMV left output rows unwritten")
    return out


def _bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def _test_single_and_mt(lib: ctypes.CDLL) -> None:
    e2m1 = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    codes = np.arange(16, dtype=np.uint8)
    packed = np.repeat((codes | (codes << 4))[:, None], 16, axis=1)
    scales = np.full((16, 1), 127, dtype=np.uint8)
    x = np.zeros(32, dtype=np.float32)
    x[0] = 1.0
    got = _gemv(lib, packed, scales, x)
    np.testing.assert_array_equal(got, e2m1)

    rng = np.random.default_rng(20260729)
    packed = rng.integers(0, 256, size=(19, 48), dtype=np.uint8)
    scales = rng.integers(120, 132, size=(19, 3), dtype=np.uint8)
    x = rng.standard_normal(96, dtype=np.float32)
    single = _gemv(lib, packed, scales, x)
    oracle = dequant_mxfp4(packed, scales).astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(single, oracle, rtol=2e-5, atol=2e-5)

    for nthreads in (1, 2, 3, 5, 16, 17, 64):
        threaded = _gemv(lib, packed, scales, x, nthreads=nthreads)
        if not np.array_equal(_bits(threaded), _bits(single)):
            raise AssertionError(f"nthreads={nthreads} differs from single GEMV")


def _test_batch(lib: ctypes.CDLL, rng: np.random.Generator) -> None:
    _bind_gemv(lib)
    lib.mxfp4_pool_init.argtypes = [ctypes.c_int]
    lib.mxfp4_pool_init.restype = ctypes.c_int
    lib.mxfp4_pool_shutdown.argtypes = []
    lib.mxfp4_pool_shutdown.restype = None
    lib.mxfp4_gemv_batch.argtypes = [
        ctypes.POINTER(U8P),
        ctypes.POINTER(U8P),
        ctypes.POINTER(F32P),
        ctypes.POINTER(F32P),
        I32P,
        I32P,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mxfp4_gemv_batch.restype = None

    specs = ((3, 32), (9, 64), (17, 96))
    packed = [
        np.ascontiguousarray(rng.integers(0, 256, size=(r, c // 2), dtype=np.uint8))
        for r, c in specs
    ]
    scales = [
        np.ascontiguousarray(rng.integers(121, 131, size=(r, c // 32), dtype=np.uint8))
        for r, c in specs
    ]
    xs = [
        np.ascontiguousarray(rng.standard_normal(c, dtype=np.float32))
        for _, c in specs
    ]
    refs = [_gemv(lib, p, s, x) for p, s, x in zip(packed, scales, xs)]
    outs = [np.full(r, np.nan, dtype=np.float32) for r, _ in specs]
    n = len(specs)
    pp = (U8P * n)(*(a.ctypes.data_as(U8P) for a in packed))
    sp = (U8P * n)(*(a.ctypes.data_as(U8P) for a in scales))
    xp = (F32P * n)(*(a.ctypes.data_as(F32P) for a in xs))
    yp = (F32P * n)(*(a.ctypes.data_as(F32P) for a in outs))
    rows = np.ascontiguousarray([r for r, _ in specs], dtype=np.int32)
    cols = np.ascontiguousarray([c for _, c in specs], dtype=np.int32)

    # 99 deliberately exceeds K3_MAX_THREADS; the public API must clamp it.
    for _ in range(2):
        for out in outs:
            out.fill(np.nan)
        lib.mxfp4_gemv_batch(
            pp, sp, xp, yp,
            rows.ctypes.data_as(I32P),
            cols.ctypes.data_as(I32P),
            n,
            99,
        )
        for i, (got, ref) in enumerate(zip(outs, refs)):
            if np.isnan(got).any() or not np.array_equal(_bits(got), _bits(ref)):
                raise AssertionError(f"batch matrix {i} differs from normal GEMV")
    lib.mxfp4_pool_shutdown()


def _test_expert_thread_bound(lib: ctypes.CDLL, rng: np.random.Generator) -> None:
    lib.mxfp4_expert_triple.argtypes = [
        U8P, U8P, U8P, U8P, U8P, U8P,
        F32P, F32P, F32P, F32P, F32P,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.mxfp4_expert_triple.restype = ctypes.c_double

    def matrix(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
        p = np.ascontiguousarray(
            rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        )
        s = np.ascontiguousarray(
            rng.integers(121, 130, size=(rows, cols // 32), dtype=np.uint8)
        )
        return p, s

    p1, s1 = matrix(3072, 3584)
    p3, s3 = matrix(3072, 3584)
    p2, s2 = matrix(3584, 3072)
    x = np.ascontiguousarray(rng.standard_normal(3584, dtype=np.float32))
    h = np.ascontiguousarray(rng.standard_normal(3072, dtype=np.float32))

    def run(nthreads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y1 = np.full(3072, np.nan, dtype=np.float32)
        y3 = np.full(3072, np.nan, dtype=np.float32)
        y2 = np.full(3584, np.nan, dtype=np.float32)
        lib.mxfp4_expert_triple(
            p1.ctypes.data_as(U8P), s1.ctypes.data_as(U8P),
            p3.ctypes.data_as(U8P), s3.ctypes.data_as(U8P),
            p2.ctypes.data_as(U8P), s2.ctypes.data_as(U8P),
            x.ctypes.data_as(F32P), h.ctypes.data_as(F32P),
            y1.ctypes.data_as(F32P), y3.ctypes.data_as(F32P), y2.ctypes.data_as(F32P),
            nthreads, 1,
        )
        if any(np.isnan(y).any() for y in (y1, y3, y2)):
            raise AssertionError("expert triple left output rows unwritten")
        return y1, y3, y2

    reference = run(4)
    over_limit = run(64)
    for name, got, expected in zip(("w1", "w3", "w2"), over_limit, reference):
        if not np.array_equal(_bits(got), _bits(expected)):
            raise AssertionError(f"expert triple {name} differs above thread limit")


def main() -> int:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    with tempfile.TemporaryDirectory(prefix="deltafin-kernel-test-") as td:
        tmp = Path(td)
        gemv_path = tmp / f"libmxfp4gemv{suffix}"
        batch_path = tmp / f"libmxfp4batch{suffix}"
        _build("fused_gemv.c", gemv_path)
        _build("fused_gemv_batch.c", batch_path)
        gemv = _load(gemv_path)
        batch = _load(batch_path)
        _bind_gemv(gemv)
        _test_single_and_mt(gemv)
        rng = np.random.default_rng(0xD37AF1)
        _test_batch(batch, rng)
        _test_expert_thread_bound(gemv, rng)

    print(
        f"PASS: {platform.system()} {platform.machine()} "
        "ABI=1, E2M1, GEMV/MT, batch, and expert thread bounds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
