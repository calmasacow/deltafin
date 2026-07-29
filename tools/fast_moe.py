"""Fast MoE expert path via the platform's fused MXFP4 dequant+GEMV library.

The native kernel is bit-exact and 41-123x the numpy-dequant + torch-matmul
path. It replaces KimiSparseMoeBlock.moe_infer math exactly: per selected
expert w1/w3 GEMV -> SiTU -> w2 GEMV -> weighted sum.
"""
import ctypes
import os
import platform

import numpy as np
import torch

try:
    import routing_record as routing_records
    from runtime_platform import (
        configured_cpu_workers,
        default_gemv_threads,
        load_native_library,
    )
except ImportError:  # imported as tools.fast_moe instead of a top-level module
    from . import routing_record as routing_records
    from .runtime_platform import (
        configured_cpu_workers,
        default_gemv_threads,
        load_native_library,
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
_REQUIRED_SYMBOLS = ("mxfp4_gemv_mt",)
_LIB, _LIB_PATH = load_native_library(
    _HERE,
    "libmxfp4gemv",
    env_var="K3_GEMV_LIB",
    required_symbols=_REQUIRED_SYMBOLS,
)
_u8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_LIB.mxfp4_gemv_mt.argtypes = [_u8, _u8, _f32, _f32,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LIB.mxfp4_gemv_mt.restype = None
_HAVE_AVX2 = getattr(_LIB, "mxfp4_have_avx2", None)
if _HAVE_AVX2 is not None:
    _HAVE_AVX2.argtypes = []
    _HAVE_AVX2.restype = ctypes.c_int

THREADS = configured_cpu_workers("K3_GEMV_THREADS", default_gemv_threads())
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0


def native_isa():
    """Describe the selected native CPU kernel without requiring a new-library ABI."""
    if _HAVE_AVX2 is not None and _HAVE_AVX2():
        return "avx2/fma"
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "neon"
    if machine in {"x86_64", "amd64"}:
        return "ssse3/fma"
    return machine or "unknown"


def _gemv(packed, scale, x):
    rows = packed.shape[0]
    y = np.empty(rows, dtype=np.float32)
    _LIB.mxfp4_gemv_mt(packed, scale, x, y, rows, packed.shape[1] * 2, THREADS)
    return y


def _situ(gate, up):
    a = SITU_BETA * np.tanh(gate / SITU_BETA) / (1.0 + np.exp(-gate))
    return a * (SITU_LINEAR_BETA * np.tanh(up / SITU_LINEAR_BETA))


def expert_ffn(raw, x):
    """raw: {w1|w2|w3: (packed, scale)}; x: fp32 [3584] contiguous -> fp32 [3584]."""
    g = _gemv(*raw["w1"], x)
    u = _gemv(*raw["w3"], x)
    h = np.ascontiguousarray(_situ(g, u), dtype=np.float32)
    return _gemv(*raw["w2"], h)


def moe_infer_fast(
        x, topk_ids, topk_weight, raw_experts, routing_record=None):
    """x: torch [N, 3584] fp32; returns torch [N, 3584] fp32.
    Mirrors KimiSparseMoeBlock.moe_infer combine: sum_k w_k * expert_k(x).
    ``routing_record`` is optional; legacy callers retain tensor conversion."""
    xnp = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
    N = xnp.shape[0]
    out = np.zeros((N, xnp.shape[1]), dtype=np.float32)
    ids, ws = routing_records.resolve(
        topk_ids, topk_weight, routing_record)
    for t in range(N):
        xt = np.ascontiguousarray(xnp[t])
        for e, w in zip(ids[t], ws[t]):
            expert_out = expert_ffn(raw_experts[e], xt)
            # The expert row has no later consumer. Scale it in place to avoid
            # a second d_model allocation for every routed expert.
            np.multiply(
                expert_out, np.float32(w), out=expert_out)
            np.add(out[t], expert_out, out=out[t])
    return torch.from_numpy(out).to(x.device, x.dtype)
