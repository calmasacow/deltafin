"""Layer-level check of the speculative capture/rollback hooks.

tools/test_spec_replay.py proves the replay MATH is exact. This proves the
PLUMBING: that `spec_decode.install()` actually intercepts a real
KimiDeltaAttention forward, records the three short-conv inputs in q,k,v order
and the recurrence kwargs, and that `rollback_replay` leaves KimiDynamicCache
holding exactly the state a shorter pass would have produced.

Runs on a miniature config with random weights — no spine, no experts, seconds.

Run:  ./venv/bin/python tools/test_spec_layer.py
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import kimi_run as kr  # noqa: E402  (installs the spec hooks on import)
import spec_decode  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)

DEV = kr.DEV
CFG = json.load(open(os.path.join(ROOT, "k3-meta/config.json")))["text_config"]
small = copy.deepcopy(CFG)
small["hidden_size"] = 256
small["num_hidden_layers"] = 2
small["linear_attn_config"] = dict(CFG["linear_attn_config"])
small["linear_attn_config"].update(num_heads=4, head_dim=32,
                                   kda_layers=[1], full_attn_layers=[2])
cfg = type(kr.config)(**small)

attn = kr.ml.KimiDeltaAttention(cfg, 0).to(DEV).eval()
for p in attn.parameters():
    p.data = torch.randn_like(p) * 0.05
attn.A_log.data = torch.randn(32, device=DEV)          # per-channel, like K3
attn.dt_bias.data = torch.randn_like(attn.dt_bias)

H = small["hidden_size"]
T = 5
kr._step_ctx["layer"] = 0
fail = 0


def fresh_cache(seed_len=3):
    c = kr.ml.KimiDynamicCache(cfg)
    attn(hidden_states=torch.randn(1, seed_len, H, device=DEV,
                                   generator=torch.Generator(DEV).manual_seed(7)),
         cache_params=c)
    return c


x = torch.randn(1, T, H, device=DEV)

# The shipped ShortConvolution has a T==1 fast path (K3_SHORTCONV=mulsum: a
# 4-tap gather+mul+sum instead of F.conv1d).  It changes q/k/v by ~1e-8, so a
# keep=1 reference taken through it can never match a batched pass bit-for-bit
# — that gap is between the T=1 and T>1 conv kernels and is already present in
# the shipped T=2 speculation, not something rollback introduces.  Pin the
# reference to the same conv kernel the batched pass used, and report the
# mulsum gap separately.
import fla.modules as _flam  # noqa: E402

for keep in range(1, T + 1):
    # reference: only `keep` positions were ever fed
    ref = fresh_cache()
    pre_rec, pre_conv = ref.recurrent_states[0], ref.conv_states[0]
    _sc, _flam.SHORTCONV = _flam.SHORTCONV, "conv1d"
    attn(hidden_states=x[:, :keep], cache_params=ref)
    _flam.SHORTCONV = _sc

    # speculative: feed all T, then roll back to `keep`
    got = kr.ml.KimiDynamicCache(cfg)
    got.recurrent_states[0], got.conv_states[0] = pre_rec, pre_conv
    spec_decode.arm()
    attn(hidden_states=x, cache_params=got)
    n_conv = len(spec_decode._CAP[0].conv)
    has_kw = spec_decode._CAP[0].fn is not None
    if keep < T:
        spec_decode.rollback_replay(got, keep, {})
    spec_decode.release()

    d_rec = (got.recurrent_states[0].float().cpu()
             - ref.recurrent_states[0].float().cpu()).abs().max().item()
    d_conv = max((got.conv_states[0][i].float().cpu()
                  - ref.conv_states[0][i].float().cpu()).abs().max().item()
                 for i in range(3))
    ok = (d_rec == 0.0 and d_conv == 0.0 and n_conv == 3 and has_kw)
    fail += not ok
    print(f"T={T} keep={keep}: captured {n_conv} convs kw={has_kw} | "
          f"recurrent maxdiff {d_rec:.3e} conv maxdiff {d_conv:.3e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

# informational: the size of the shipped T=1 conv fast-path gap, for scale
alt = fresh_cache()
pre_rec, pre_conv = alt.recurrent_states[0], alt.conv_states[0]
attn(hidden_states=x[:, :1], cache_params=alt)          # mulsum (shipped T=1)
ref1 = kr.ml.KimiDynamicCache(cfg)
ref1.recurrent_states[0], ref1.conv_states[0] = pre_rec, pre_conv
_sc, _flam.SHORTCONV = _flam.SHORTCONV, "conv1d"
attn(hidden_states=x[:, :1], cache_params=ref1)
_flam.SHORTCONV = _sc
print(f"[info] shipped T=1 mulsum vs conv1d state gap: "
      f"{(alt.recurrent_states[0] - ref1.recurrent_states[0]).abs().max():.3e} "
      f"(pre-existing; also present in the shipped T=2 speculation)", flush=True)

# capture disarmed must leave the kernels byte-for-byte on the shipped path
c1, c2 = fresh_cache(), fresh_cache()
attn(hidden_states=x, cache_params=c1)
attn(hidden_states=x, cache_params=c2)
d = (c1.recurrent_states[0] - c2.recurrent_states[0]).abs().max().item()
print(f"disarmed determinism: maxdiff {d:.3e} {'PASS' if d == 0.0 else 'FAIL'}",
      flush=True)
fail += d != 0.0

print("SPEC LAYER:", "PASS" if fail == 0 else f"FAIL ({fail})", flush=True)
sys.exit(1 if fail else 0)
