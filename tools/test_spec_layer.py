"""Layer-level check of the speculative capture/rollback hooks.

tools/test_spec_replay.py proves the replay MATH is exact. This proves the
PLUMBING: that `spec_decode.install()` actually intercepts a real
KimiDeltaAttention forward, records the three short-conv inputs in q,k,v order
and the recurrence kwargs, and that `rollback_replay` leaves KimiDynamicCache
holding the numerically equivalent state a shorter pass would have produced.

The replay of captured tensors is bit-exact (test_spec_replay.py). A separate
shorter forward can differ by a few ulps because dense projection kernels are
shape-dependent; that cross-shape numerical gate is deliberately explicit
below and full generation must still emit the control token sequence.

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
REC_ATOL = 1e-6
CONV_ATOL = 2e-6


def fresh_cache(seed_len=3):
    c = kr.ml.KimiDynamicCache(cfg)
    attn(hidden_states=torch.randn(1, seed_len, H, device=DEV,
                                   generator=torch.Generator(DEV).manual_seed(7)),
         cache_params=c)
    return c


x = torch.randn(1, T, H, device=DEV)

for keep in range(1, T + 1):
    # Reference: only `keep` positions were ever fed. The portable mulsum
    # short-convolution itself uses identical W-wide reductions for T=1 and
    # T>1; the surrounding dense projections may still choose a different
    # shape-dependent reduction.
    ref = fresh_cache()
    pre_rec, pre_conv = ref.recurrent_states[0], ref.conv_states[0]
    attn(hidden_states=x[:, :keep], cache_params=ref)

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
    ok = (
        d_rec <= REC_ATOL
        and d_conv <= CONV_ATOL
        and n_conv == 3
        and has_kw
    )
    fail += not ok
    print(f"T={T} keep={keep}: captured {n_conv} convs kw={has_kw} | "
          f"recurrent maxdiff {d_rec:.3e} conv maxdiff {d_conv:.3e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

# A batched production call must remain numerically equivalent to T ordinary
# decode calls without invoking rollback at all. The isolated short-conv test
# is bit-exact; this full-layer comparison includes shape-dependent linears.
batched = fresh_cache()
sequential = fresh_cache()
attn(hidden_states=x, cache_params=batched)
for index in range(T):
    attn(hidden_states=x[:, index:index + 1], cache_params=sequential)
d_rec = (
    batched.recurrent_states[0] - sequential.recurrent_states[0]
).abs().max().item()
d_conv = max(
    (
        batched.conv_states[0][index] - sequential.conv_states[0][index]
    ).abs().max().item()
    for index in range(3)
)
batch_ok = d_rec <= REC_ATOL and d_conv <= CONV_ATOL
print(
    f"batched versus sequential: recurrent maxdiff {d_rec:.3e} "
    f"conv maxdiff {d_conv:.3e} {'PASS' if batch_ok else 'FAIL'}",
    flush=True,
)
fail += not batch_ok

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
