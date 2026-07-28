import sys, os, json, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT+'/tools'); # package import via tools/k3pkg
torch.set_grad_enabled(False); torch.manual_seed(0)
import importlib
from k3pkg import modeling_kimi_linear as ml
cfg = json.load(open(ROOT+'/k3-meta/config.json'))['text_config']
# shrink dims for a fast math test
cfg = dict(cfg, hidden_size=64, num_hidden_layers=4,
           linear_attn_config=dict(cfg['linear_attn_config'], num_heads=4, head_dim=16))
Cfg = getattr(ml, 'KimiLinearConfig', None) or importlib.import_module('k3pkg.configuration_kimi_k3').KimiLinearConfig
config = Cfg(**cfg); config._attn_implementation='eager'
att = ml.KimiDeltaAttention(config, layer_idx=1).float().eval()
for p in att.parameters(): p.data = torch.randn_like(p.data)*0.05
# per-channel A_log like the real checkpoint ([head_dim])
att.A_log = torch.nn.Parameter(torch.randn(16)*0.1)
x = torch.randn(1, 9, 64)

c_all = ml.KimiDynamicCache(config)
o_all = att(x, cache_params=c_all)                      # one chunk of 9

c_mix = ml.KimiDynamicCache(config)
o_pre = att(x[:, :6], cache_params=c_mix)               # chunk prefill of 6
outs = [o_pre]
for t in range(6, 9):
    outs.append(att(x[:, t:t+1], cache_params=c_mix))   # stepwise decode
o_mix = torch.cat(outs, dim=1)

diff = (o_all - o_mix).abs().max().item()
sdiff = (c_all.recurrent_states[1] - c_mix.recurrent_states[1]).abs().max().item()
print(f"output maxdiff chunk-vs-stepwise: {diff:.3e}")
print(f"state  maxdiff:                   {sdiff:.3e}")
assert diff < 1e-4 and sdiff < 1e-4, "MISMATCH"
print("KDA shim self-consistency: PASS")
