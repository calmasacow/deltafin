#!/usr/bin/env python3
"""Full-spine int8 conversion: every big 2-D bf16 Linear in k3-resident/tensors
(plus lm_head) -> k3-resident-int8/tensors/<name>.i8 (int8 row-major) + <name>.sc
(fp16 per-row scales). Norms/convs/A_log/dt_bias/res_projs/bias stay bf16 (loader
falls back to the original dir). Resumable; sequential (memory-light)."""
import json, os, sys, time
import numpy as np

K3 = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(K3, "k3-resident/tensors")
OUT = os.path.join(K3, "k3-resident-int8/tensors")
INV = json.load(open(os.path.join(K3, "k3-meta/tensor_inventory_offsets.json")))
os.makedirs(OUT, exist_ok=True)

EXCLUDE = ("conv1d", "norm", "A_log", "dt_bias", "res_proj",
           "e_score_correction_bias", ".experts.", "vision", "mm_projector")

targets = []
for name, t in INV.items():
    if t["dtype"] != "BF16" or len(t["shape"]) != 2:
        continue
    if any(x in name for x in EXCLUDE):
        continue
    if not os.path.exists(os.path.join(RES, name)):
        continue
    targets.append((name, t["shape"]))
targets.sort()
total = sum(r * c * 2 for _, (r, c) in targets)
print(f"targets: {len(targets)} tensors, {total/1e9:.1f} GB bf16 -> ~{total/2/1e9 + sum(r*2 for _,(r,c) in targets)/1e9:.1f} GB int8+scales", flush=True)

t0, done = time.time(), 0
for i, (name, (rows, cols)) in enumerate(targets):
    op, os_ = os.path.join(OUT, name + ".i8"), os.path.join(OUT, name + ".sc")
    if os.path.exists(op) and os.path.getsize(op) == rows * cols:
        done += rows * cols * 2
        continue
    buf = np.fromfile(os.path.join(RES, name), dtype=np.uint16).reshape(rows, cols)
    w = (buf.astype(np.uint32) << 16).view(np.float32)          # bf16 -> fp32
    sc = np.abs(w).max(axis=1, keepdims=True) / 127.0
    sc[sc == 0] = 1.0
    q = np.clip(np.rint(w / sc), -127, 127).astype(np.int8)
    q.tofile(op + ".tmp"); os.replace(op + ".tmp", op)
    sc.astype(np.float16).tofile(os_ + ".tmp"); os.replace(os_ + ".tmp", os_)
    done += rows * cols * 2
    if i % 100 == 0:
        el = time.time() - t0
        print(f"{i}/{len(targets)} {done/1e9:.1f}/{total/1e9:.1f} GB "
              f"({done/1e9/max(el,1):.2f} GB/s, eta {(total-done)/1e9/max(done/1e9/max(el,1),0.01)/60:.0f} min)", flush=True)
print(f"DONE {done/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min", flush=True)
