import sys, os, torch, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["K3_FAST_MOE"] = "0"   # import kimi_run with slow path available
import kimi_run as kr
import k3loader, fast_moe

torch.manual_seed(7)
li, ids = 1, [0, 1, 2, 3]                     # cached since the smoke test
x = torch.randn(3, 3584) * 0.5                # latent-space inputs
topk_ids = torch.tensor([[0, 1], [2, 3], [1, 2]])
topk_w = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)

# reference: Moonshot module path (dequant -> nn.Linear -> SituAndMul)
deq = k3loader.fetch_experts(li, ids, dequant=True)
with torch.device('meta'):
    block = kr.ml.KimiSparseMoeBlock(kr.config).eval()
for e in ids:
    for wn in ("w1", "w2", "w3"):
        kr.set_param(block.experts[e], wn + ".weight", deq[e][wn])
kr._step_ctx["layer"] = li
t0 = time.time(); ref = kr._orig_moe_infer(block, x, topk_ids, topk_w); t_ref = time.time()-t0

raw = k3loader.fetch_experts(li, ids, dequant=False)
t0 = time.time(); fast = fast_moe.moe_infer_fast(x, topk_ids, topk_w, raw); t_fast = time.time()-t0

rel = (ref - fast).abs().max() / ref.abs().max()
print(f"ref path {t_ref*1e3:.0f}ms | fast path {t_fast*1e3:.0f}ms | speedup {t_ref/t_fast:.0f}x")
print(f"max abs diff {(ref-fast).abs().max():.3e} | max rel {rel:.3e}")
assert rel < 1e-4, "MISMATCH"
print("fast MoE path: VALIDATED")
