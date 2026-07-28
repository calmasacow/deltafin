#!/usr/bin/env python3
"""quantize_spine.py -- int8 symmetric per-row quantization of the K3 resident spine.

For each big Linear weight (bf16 [rows, cols]):
  scale[r] = absmax(row_r) / 127   (fp32 -> stored fp16)
  q[r]     = clip(round(w[r] / scale[r]), -127, 127)  int8
Blobs written as <name>.i8 (int8 payload, row-major) + <name>.sc (fp16 scales).
A manifest.json records shapes + per-tensor weight-space reconstruction error.

Selection rule (per experiment spec): 2-D bf16 Linears under self_attn.* (excluding
conv1d), shared_experts.*, routed_expert_{up,down}_proj, block_sparse_moe.gate.weight.
Norms / convs / A_log / dt_bias / res_projs stay bf16.

Usage: quantize_spine.py [--layers 3 5] [--copy-first]
Reads copies from ./bf16/ (made with --copy-first from k3-resident, READ ONLY source).
"""
import argparse, json, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K3 = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(K3, "k3-resident/tensors")
INV = json.load(open(os.path.join(K3, "k3-meta/tensor_inventory_offsets.json")))
PFX = "language_model.model."

EXCLUDE_SUBSTR = ("conv1d", "norm", "A_log", "dt_bias", "res_proj",
                  "e_score_correction_bias", ".experts.")


def is_quant_target(name, meta):
    if not name.endswith(".weight"):
        return False
    if any(s in name for s in EXCLUDE_SUBSTR):
        return False
    if meta["dtype"] != "BF16" or len(meta["shape"]) != 2:
        return False
    ok = (".self_attn." in name
          or ".shared_experts." in name
          or ".routed_expert_up_proj." in name
          or ".routed_expert_down_proj." in name
          or name.endswith(".block_sparse_moe.gate.weight"))
    return ok


def layer_targets(layer):
    pfx = f"{PFX}layers.{layer}."
    return sorted(n for n, m in INV.items()
                  if n.startswith(pfx) and is_quant_target(n, m))


def load_bf16_as_f32(path, shape):
    """Read a raw bf16 blob -> fp32 numpy (bf16 = upper 16 bits of fp32)."""
    raw = np.fromfile(path, dtype=np.uint16).reshape(shape)
    f32 = np.zeros(shape, dtype=np.uint32)
    f32 |= raw.astype(np.uint32) << 16
    return f32.view(np.float32)


def quantize_rowwise_int8(w):
    """w: fp32 [rows, cols] -> (int8 q, fp16 scales). Scale applied as stored fp16."""
    absmax = np.abs(w).max(axis=1)
    scale16 = (absmax / 127.0).astype(np.float16)
    scale16[scale16 == 0] = np.float16(6e-8)          # guard all-zero rows
    s = scale16.astype(np.float32)[:, None]
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return q, scale16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--copy-first", action="store_true",
                    help="cp resident blobs into ./bf16/ before quantizing")
    args = ap.parse_args()

    bf16dir, i8dir = os.path.join(HERE, "bf16"), os.path.join(HERE, "int8")
    os.makedirs(bf16dir, exist_ok=True)
    os.makedirs(i8dir, exist_ok=True)

    manifest, tot_bf16, tot_i8 = {}, 0, 0
    for L in args.layers:
        for name in layer_targets(L):
            meta = INV[name]
            shape = meta["shape"]
            src = os.path.join(RES, name)
            cp = os.path.join(bf16dir, name)
            if args.copy_first and not os.path.exists(cp):
                shutil.copyfile(src, cp)          # work on copies, source READ ONLY
            t0 = time.time()
            w = load_bf16_as_f32(cp, shape)
            q, s16 = quantize_rowwise_int8(w)
            # weight-space reconstruction error (vs the fp32 the driver computes in)
            deq = q.astype(np.float32) * s16.astype(np.float32)[:, None]
            err = deq - w
            rel_fro = float(np.linalg.norm(err) / (np.linalg.norm(w) + 1e-30))
            max_abs = float(np.abs(err).max())
            q.tofile(os.path.join(i8dir, name + ".i8"))
            s16.tofile(os.path.join(i8dir, name + ".sc"))
            bb, ib = q.size * 2, q.size + s16.size * 2
            tot_bf16 += bb
            tot_i8 += ib
            manifest[name] = {"shape": shape, "bf16_bytes": bb, "int8_bytes": ib,
                              "rel_fro_err_w": rel_fro, "max_abs_err_w": max_abs,
                              "quant_s": round(time.time() - t0, 2)}
            print(f"  {name.split('layers.')[-1]:60s} {str(shape):16s} "
                  f"rel_fro={rel_fro:.2e} {bb/1e6:7.1f}MB -> {ib/1e6:7.1f}MB",
                  flush=True)
            del w, q, deq, err
    manifest["_totals"] = {"bf16_bytes": tot_bf16, "int8_bytes": tot_i8,
                           "ratio": tot_bf16 / tot_i8}
    json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    print(f"\nTOTAL {tot_bf16/1e6:.1f}MB bf16 -> {tot_i8/1e6:.1f}MB int8 "
          f"(x{tot_bf16/tot_i8:.3f} smaller)")


if __name__ == "__main__":
    main()
