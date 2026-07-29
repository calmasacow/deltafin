"""One decode token (T=1) through all 93 layers: real template layers, real
shapes, random weights, routed experts stubbed out. Isolates the
"attention + norms" bucket with no disk I/O and no expert fetch, so the flag
A/B takes ~2 s of GPU work instead of a 16 s token.

    python tools/bench_attn_compute.py             # env flags select the path
"""
import os, sys, json, time, importlib
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/tools")
import torch, torch.nn as nn
torch.set_grad_enabled(False)
torch.set_num_threads(8)
from k3pkg import modeling_kimi_linear as ml
CFG = json.load(open(ROOT + "/k3-meta/config.json"))["text_config"]
Cfg = getattr(ml, "KimiLinearConfig", None) or importlib.import_module(
    "k3pkg.configuration_kimi_k3").KimiLinearConfig
config = Cfg(**CFG); config._attn_implementation = "eager"
DEV = torch.device("mps")
H = config.hidden_size
NL = config.num_hidden_layers

import attn_fast                                   # noqa: E402
attn_fast.install(ml)
print("[attn_fast]", attn_fast.describe(DEV), flush=True)


def set_param(root, dotted, tensor):
    obj = root
    for p in dotted.split(".")[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, dotted.split(".")[-1], nn.Parameter(tensor, requires_grad=False))


def fill(mod):
    for n, p in list(mod.named_parameters()):
        if ".experts." in n:
            continue
        set_param(mod, n, torch.randn(p.shape, device=DEV, dtype=torch.float32) * 0.02)
    return mod


# stub the routed experts entirely
ml.KimiSparseMoeBlock.moe_infer = lambda self, x, i, w: torch.zeros_like(x)

with torch.device("meta"):
    l0 = ml.KimiDecoderLayer(config, 0).eval()
    tpl_kda = ml.KimiDecoderLayer(config, 1).eval()
    tpl_mla = ml.KimiDecoderLayer(config, 3).eval()
layers = [l0 if i == 0 else (tpl_kda if config.is_kda_layer(i) else tpl_mla)
          for i in range(NL)]
for m in (l0, tpl_kda, tpl_mla):
    fill(m)
print("[built]", flush=True)

cache = ml.KimiDynamicCache(config)
CTX = 5
for i in range(NL):
    if config.is_kda_layer(i):
        D = 96 * 128
        cache.conv_states[i] = tuple(torch.randn(1, D, 4, device=DEV) * 0.02 for _ in range(3))
        cache.recurrent_states[i] = torch.randn(1, 96, 128, 128, device=DEV) * 0.01
        if attn_fast.RECUR == "cpu":
            cache.recurrent_states[i] = cache.recurrent_states[i].cpu()
    else:
        cache.key_cache[i] = torch.randn(1, 96, CTX, 192, device=DEV) * 0.02
        cache.value_cache[i] = torch.randn(1, 96, CTX, 128, device=DEV) * 0.02
state0 = ([c for c in cache.conv_states], [r for r in cache.recurrent_states],
          [k for k in cache.key_cache], [v for v in cache.value_cache])


def one_token():
    for i in range(NL):
        cache.conv_states[i] = state0[0][i]
        cache.recurrent_states[i] = state0[1][i]
        cache.key_cache[i] = state0[2][i]
        cache.value_cache[i] = state0[3][i]
    hidden = torch.randn(1, 1, H, device=DEV) * 0.02
    br = hidden.new_zeros(1, 0, H)
    kda_t = mla_t = 0.0
    for i, layer in enumerate(layers):
        layer.layer_idx = i
        layer.self_attn.layer_idx = i
        t0 = time.perf_counter()
        hidden, br = layer(hidden, attention_mask=None, position_ids=None,
                           past_key_values=cache, use_cache=True, block_residual=br)
        if PERLAYER:
            torch.mps.synchronize()
            dt = time.perf_counter() - t0
            if layer.is_linear_attn:
                kda_t += dt
            else:
                mla_t += dt
    torch.mps.synchronize()
    return kda_t, mla_t


PERLAYER = os.environ.get("PERLAYER", "1") == "1"
one_token()   # warm
R = int(os.environ.get("R", "3"))
tot = []
for _ in range(R):
    t0 = time.perf_counter()
    kda_t, mla_t = one_token()
    tot.append(time.perf_counter() - t0)
tot.sort()
print(f"RESULT token(93 layers, no experts): {tot[len(tot)//2]*1e3:.0f} ms  "
      f"(all: {[round(x*1e3) for x in tot]})  "
      f"kda {kda_t*1e3:.0f} ms/69  mla {mla_t*1e3:.0f} ms/24", flush=True)
