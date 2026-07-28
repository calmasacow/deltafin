"""Rollback validation: a spec pass with a deliberately WRONG draft must, after
restore_states + redo, leave identical next-token and near-identical states vs a
straight decode. Also validates causal isolation (position 0 logits unaffected
by the garbage draft at position 1)."""
import os, sys
os.environ.setdefault("K3_SPINE", "int8"); os.environ.setdefault("K3_DEV", "mps")
os.environ["K3_PIN_LAYERS"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, kimi_run as kr

IDS = [1008, 10484, 318, 15383, 387]
layers = kr.build_layers(); embed = kr.LazyEmbed()

cacheA = kr.ml.KimiDynamicCache(kr.config)
lg = kr.forward_pass(layers, cacheA, embed(IDS), 0, verbose=False)
n1 = int(lg[0, -1].argmax())
lgA = kr.forward_pass(layers, cacheA, embed([n1]), 1, verbose=False)
n2A = int(lgA[0, -1].argmax())
print(f"straight path: n1={n1} n2={n2A}", flush=True)

cacheB = kr.ml.KimiDynamicCache(kr.config)
kr.forward_pass(layers, cacheB, embed(IDS), 0, verbose=False)
snap = kr.snapshot_states(cacheB)
WRONG = 424242 % kr.config.vocab_size
lgB = kr.forward_pass(layers, cacheB, embed([n1, WRONG]), 1, verbose=False)
v1 = int(lgB[0, 0].argmax())
print(f"causal isolation: pos-0 argmax with garbage draft = {v1} (want {n2A}):",
      "PASS" if v1 == n2A else "FAIL", flush=True)
kr.restore_states(cacheB, snap)
lgB2 = kr.forward_pass(layers, cacheB, embed([n1]), 1, verbose=False)
n2B = int(lgB2[0, -1].argmax())
sdiff = max((cacheA.recurrent_states[i] - cacheB.recurrent_states[i]).abs().max().item()
            for i in range(kr.NL) if cacheA.recurrent_states[i] is not None)
ldiff = (lgA[0, -1].float().cpu() - lgB2[0, -1].float().cpu()).abs().max().item()
print(f"post-rollback: n2={n2B} (want {n2A}) | state maxdiff {sdiff:.2e} | logit maxdiff {ldiff:.2e}", flush=True)
assert v1 == n2A and n2B == n2A and sdiff < 1e-4, "ROLLBACK FAIL"
print("ROLLBACK: PASS", flush=True)
