"""Crux test for multi-token speculation: partial-accept rollback is EXACT.

No model weights needed — this exercises the two pieces of state a speculative
pass advances and that a partial accept has to unwind:

  1. the KDA recurrent state (a fold; not truncatable, not invertible)
  2. the KDA short-conv state (a rolling window of raw inputs)

For each we compare
    "run T columns, then roll back to n"   vs   "run only n columns"
at the real K3 shapes, and demand a bit-identical result (max diff 0.0), which
is what makes an accepted prefix reproduce the K3_SPEC=0 sequence exactly.

Run:  ./venv/bin/python tools/test_spec_replay.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import spec_decode  # noqa: E402
from fla.modules import ShortConvolution  # noqa: E402
from fla.ops.kda import chunk_kda  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)

# real K3 KDA shapes
B, H, K, V, W = 1, 96, 128, 128, 4
DPROJ = H * K                      # 12288
T = 5                              # K3_SPEC_DEPTH=4 -> D+1 = 5 positions

DEVS = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
fail = 0


def kda_kwargs(dev, t):
    return dict(
        q=torch.randn(B, t, H, K, device=dev),
        k=torch.randn(B, t, H, K, device=dev),
        v=torch.randn(B, t, H, V, device=dev),
        g=torch.randn(B, t, H, K, device=dev),
        beta=torch.randn(B, t, H, device=dev),
        A_log=torch.randn(K, device=dev),
        dt_bias=torch.randn(H * K, device=dev),
        initial_state=torch.randn(B, H, K, V, device=dev),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
    )


for dev in DEVS:
    kw = kda_kwargs(dev, T)

    # --- 1. recurrent state -------------------------------------------------
    for n in range(1, T):
        ref_kw = dict(kw)
        for name in ("q", "k", "v", "g", "beta"):
            ref_kw[name] = kw[name][:, :n]
        _, S_ref = chunk_kda(**ref_kw)               # "only n tokens were fed"

        rec = spec_decode._Rec()
        rec.fn, rec.kw = chunk_kda, dict(kw)         # what capture() records
        rec.conv = []
        S, _ = spec_decode._replay_layer(rec, n)     # roll a T-pass back to n

        d = (S.float() - S_ref.float()).abs().max().item()
        ok = d == 0.0
        fail += not ok
        print(f"[{dev}] recurrent state T={T} -> keep {n}: maxdiff {d:.3e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    # --- 2. short-conv state ------------------------------------------------
    conv = ShortConvolution(DPROJ, W, activation="silu").to(dev)
    x = torch.randn(B, T, DPROJ, device=dev)
    pre = torch.randn(B, DPROJ, W, device=dev)

    _, full_state = conv(x=x, cache=pre, output_final_state=True)
    for n in range(1, T):
        _, ref_state = conv(x=x[:, :n], cache=pre, output_final_state=True)
        rec = spec_decode._Rec()
        rec.fn, rec.kw = chunk_kda, dict(kw)
        rec.conv = [(x, pre)]
        _, convs = spec_decode._replay_layer(rec, n)
        d = (convs[0].float() - ref_state.float()).abs().max().item()
        ok = d == 0.0
        fail += not ok
        print(f"[{dev}] conv state      T={T} -> keep {n}: maxdiff {d:.3e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    rec = spec_decode._Rec()
    rec.fn, rec.kw, rec.conv = chunk_kda, dict(kw), [(x, pre)]
    d = (spec_decode._replay_layer(rec, T)[1][0].float()
         - full_state.float()).abs().max().item()
    ok = d == 0.0
    fail += not ok
    print(f"[{dev}] conv state      T={T} -> keep {T}: maxdiff {d:.3e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)


# --- 3. K3_SPEC_RECUR_MATCH: the T>4 batch must take the CPU branch a T=1
#        decode would have taken under K3_KDA_RECUR=cpu, and replay must follow
#        it there (otherwise a near-tie token could flip vs K3_SPEC=0).
if "mps" in DEVS:
    kw = kda_kwargs("mps", T)
    cpu_kw = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in kw.items()}
    _, S_cpu = chunk_kda(**cpu_kw)
    spec_decode._FORCE_CPU = True
    o, S = spec_decode._run(chunk_kda, dict(kw))
    ok = (S.device.type == "cpu" and o.device.type == "mps"
          and (S - S_cpu).abs().max().item() == 0.0)
    fail += not ok
    print(f"[match] forced CPU branch at T={T}: state on {S.device.type}, "
          f"out on {o.device.type}, maxdiff vs CPU run "
          f"{(S - S_cpu).abs().max().item():.3e} {'PASS' if ok else 'FAIL'}",
          flush=True)
    for n in (1, 3):
        ref_kw = dict(cpu_kw)
        for name in ("q", "k", "v", "g", "beta"):
            ref_kw[name] = cpu_kw[name][:, :n]
        _, S_ref = chunk_kda(**ref_kw)
        rec = spec_decode._Rec()
        rec.fn, rec.kw, rec.conv = chunk_kda, dict(kw), []
        S, _ = spec_decode._replay_layer(rec, n)
        d = (S.cpu() - S_ref).abs().max().item()
        ok = d == 0.0
        fail += not ok
        print(f"[match] forced-CPU replay T={T} -> keep {n}: maxdiff {d:.3e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    spec_decode._FORCE_CPU = False


# --- 4. accept-prefix arithmetic (pure python, no tensors) -------------------
def accept(drafts, argm):
    k = 0
    while k < len(drafts) and argm[k] == drafts[k]:
        k += 1
    return k, argm[:k + 1]


CASES = [
    # drafts,        model argmax per position,  expected (k, emitted)
    ([7, 8, 9, 10], [7, 8, 9, 10, 11], (4, [7, 8, 9, 10, 11])),   # full accept
    ([7, 8, 9, 10], [7, 8, 99, 0, 0], (2, [7, 8, 99])),           # partial
    ([7, 8, 9, 10], [42, 0, 0, 0, 0], (0, [42])),                 # total miss
    ([7], [7, 8], (1, [7, 8])),                                   # depth-1 hit
]
for drafts, argm, want in CASES:
    got = accept(drafts, argm)
    ok = got == want
    fail += not ok
    print(f"[logic] drafts={drafts} argm={argm} -> {got} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

print("SPEC REPLAY:", "PASS" if fail == 0 else f"FAIL ({fail})", flush=True)
sys.exit(1 if fail else 0)
