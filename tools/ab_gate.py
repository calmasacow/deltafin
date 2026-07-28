"""A/B gate: one prefill pass of the standard prompt under K3_DEV/K3_SPINE config.
Prints a JSON line with argmax + top-5 for cross-config comparison."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import kimi_run as kr

IDS = [1008, 10484, 318, 15383, 387]   # "The capital of France is"
t0 = time.time()
layers = kr.build_layers()
cache = kr.ml.KimiDynamicCache(kr.config)
kr._step_ctx["step"] = 0
logits = kr.forward_pass(layers, cache, kr.LazyEmbed()(IDS), step=0, verbose=False)
lg = logits[0, -1].float().cpu()
top = torch.topk(lg, 5)
print("ABRESULT " + json.dumps({
    "dev": str(kr.DEV), "spine": kr.SPINE, "wall_s": round(time.time()-t0, 1),
    "argmax": int(lg.argmax()),
    "top5_ids": top.indices.tolist(),
    "top5_logits": [round(v, 4) for v in top.values.tolist()],
    "times": {k: round(v, 1) for k, v in kr.TIMES.items()},
}), flush=True)
