"""Fast MoE expert path: fused MXFP4 dequant+GEMV via libmxfp4gemv.dylib (bit-exact
kernel, 41-123x the numpy-dequant + torch-matmul path). Replaces KimiSparseMoeBlock.moe_infer
math exactly: per selected expert w1/w3 GEMV -> SiTU -> w2 GEMV -> weighted sum."""
import ctypes, os
import numpy as np
import torch

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmxfp4gemv.dylib"))
_u8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_LIB.mxfp4_gemv_mt.argtypes = [_u8, _u8, _f32, _f32,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LIB.mxfp4_gemv_mt.restype = None

THREADS = int(os.environ.get("K3_GEMV_THREADS", "4"))
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0


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


def moe_infer_fast(x, topk_ids, topk_weight, raw_experts):
    """x: torch [N, 3584] fp32; returns torch [N, 3584] fp32.
    Mirrors KimiSparseMoeBlock.moe_infer combine: sum_k w_k * expert_k(x)."""
    xnp = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
    N = xnp.shape[0]
    out = np.zeros((N, xnp.shape[1]), dtype=np.float32)
    ids = topk_ids.tolist()
    ws = topk_weight.to(torch.float32).tolist()
    for t in range(N):
        xt = np.ascontiguousarray(xnp[t])
        for e, w in zip(ids[t], ws[t]):
            out[t] += np.float32(w) * expert_ffn(raw_experts[e], xt)
    return torch.from_numpy(out).to(x.device, x.dtype)
