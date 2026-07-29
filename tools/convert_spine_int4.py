#!/usr/bin/env python3
"""Spine quantization: uniform int4 (original mode) or a MIXED-PRECISION spine.

MIXED MODE (--policy) writes a per-ROLE mix of i4 / i6 / i8 into ONE directory.
Motivation and measurements: tools/spine_sensitivity.py. Short version — no role
is cheap at 4 bits (every role lands at 0.108 +/- 0.005 relative Frobenius,
11.6x int8, because the weights are near-Gaussian and 4-bit absmax is already at
the scalar rate-distortion floor), so a mixed spine is a pure "how much of the
model do we let go to 10% error" trade. The i6 codec (0.78 B/elt, 2.6x int8
error) sits on a much better point of that frontier and is available to policies
here as `i6`.

  python tools/convert_spine_int4.py --policy moe4 --dry-run
  python tools/convert_spine_int4.py --policy moe4        # ship-candidate mix
  python tools/convert_spine_int4.py --policy moe4 --layers 40-51 --verify
  python tools/convert_spine_int4.py --policy-roles "shared_experts=i6,lm_head=i8"

Only the tensors the policy assigns a non-i8 codec are written; the loader
(tools/mixed_spine.py) falls back to the existing int8 spine for everything
else, so a mixed directory costs only the bytes it actually changes. Pass
--int8-mode link|copy to materialize a self-contained directory instead.

ORIGINAL MODE (no --policy) is unchanged, byte for byte:
full-spine int4 conversion (OPT-IN, QUALITY-GATED — never a default).

Group-wise symmetric int4 of exactly the tensor set convert_spine_int8.py picks
(big 2-D bf16 Linears; norms / convs / A_log / dt_bias / res_projs / router bias /
vision / routed experts all excluded — the loader falls back to bf16 or int8 for
those). Output goes to k3-resident-int4/tensors/:

  <name>.i4  uint8 [rows, cw/2]  two int4 nibbles per byte, LOW nibble = even col
                                 (same nibble order as the MXFP4 expert format)
  <name>.s4  float16 [rows, ng]  one scale per group of `g` consecutive columns
  ../meta.json                   {format, version, group, ...} — the loader reads it

Quantizer, per group of g consecutive elements in a row:
  s   = fp16( absmax / 7 )          # rounded to fp16 FIRST, then used, so the
  q   = clip(round(w / s), -7, 7)   # dequant is exactly reproducible on load
Values are 2's-complement 4-bit; -8 is never emitted (symmetric range).

Size at g=64: 0.5 + 2/64 = 0.53125 B/elt  ->  113.5 GB bf16 becomes ~30.1 GB
(int8 is ~56.7 GB). Weight error is ~3x int8's; see tools/int4_quality_report.py.

Resumable (skips complete outputs), sequential and memory-light (row chunks).

  python tools/convert_spine_int4.py                 # full spine, g=64
  python tools/convert_spine_int4.py --sample 8      # 8 tensors, with error stats
  python tools/convert_spine_int4.py --group 32      # finer groups (bigger, better)
  python tools/convert_spine_int4.py --verify        # error stats on every tensor
"""
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spine_codec  # noqa: E402

K3 = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(K3, "k3-resident/tensors")
INT8 = os.path.join(K3, "k3-resident-int8/tensors")
INV = json.load(open(os.path.join(K3, "k3-meta/tensor_inventory_offsets.json")))

# Identical selection rule to convert_spine_int8.py — keep these in sync.
EXCLUDE = ("conv1d", "norm", "A_log", "dt_bias", "res_proj",
           "e_score_correction_bias", ".experts.", "vision", "mm_projector")

FORMAT = "int4-sym-group"
VERSION = 1

# --------------------------------------------------------------- policies ---
# A policy maps a substring of the ROLE (the tensor name with "layers.N." and
# the "language_model." prefixes stripped) to a codec. First match wins;
# anything unmatched stays i8, i.e. is left to the existing int8 spine.
#
# Roles and their share of the 56.7 GB int8 spine:
#   attention   g_proj 8.2 / o_proj 8.2 / q,k,v 6.1 each / q_a,q_b,kv_a,kv_b 1.3   64%
#   shared MLP  gate 4.1 / up 4.1 / down 4.1                                       21%
#   latent MoE  routed_expert_down 2.4 / routed_expert_up 2.4                       8%
#   lm_head 1.2 / embed 1.2 (never read per token) / router gate 0.6 / dense mlp 0.7
POLICIES = {
    # int4 on the MoE-side projections only: the shared-expert MLP and the two
    # latent projections that bracket the routed experts (which the checkpoint
    # already ships as 4-bit MXFP4). Attention, router and lm_head stay int8.
    "moe4": [("shared_experts.", "i4"), ("routed_expert_down_proj", "i4"),
             ("routed_expert_up_proj", "i4")],
    # the same set at 6 bits — 2.6x int8 error instead of 11.6x, for 0.78 B/elt
    "moe6": [("shared_experts.", "i6"), ("routed_expert_down_proj", "i6"),
             ("routed_expert_up_proj", "i6")],
    # the aggressive mix the sensitivity study points at: i4 where the branch is
    # already 4-bit or gated, i6 on the attention bulk, i8 on the two tensors
    # whose error lands directly on token identity (router scores, lm_head).
    "mix46": [("block_sparse_moe.gate.weight", "i8"), ("lm_head", "i8"),
              ("shared_experts.", "i4"), ("routed_expert_down_proj", "i4"),
              ("routed_expert_up_proj", "i4"), ("", "i6")],
    # everything at 6 bits except router + lm_head (56.7 -> 44.6 GB)
    "all6": [("block_sparse_moe.gate.weight", "i8"), ("lm_head", "i8"), ("", "i6")],
    # everything at 4 bits except router + lm_head — the already-rejected point,
    # kept so the gate can be re-run against it
    "all4": [("block_sparse_moe.gate.weight", "i8"), ("lm_head", "i8"), ("", "i4")],
}


def role_of(name):
    return re.sub(r"layers\.\d+\.", "", name).replace("language_model.model.", "") \
             .replace("language_model.", "")


def parse_policy(spec):
    """'moe4' or 'role_substr=codec,...' -> [(substr, codec)]"""
    if spec in POLICIES:
        return POLICIES[spec]
    if spec.startswith("@"):
        return [tuple(x) for x in json.load(open(spec[1:]))]
    rules = []
    for part in spec.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        if v not in spine_codec.CODECS:
            sys.exit(f"bad codec {v!r} in --policy {spec!r}; "
                     f"expected one of {spine_codec.CODECS}")
        rules.append((k.strip(), v))
    if not rules:
        sys.exit(f"empty policy {spec!r}")
    return rules


def codec_for(name, rules):
    r = role_of(name)
    for pat, codec in rules:
        if pat in r:
            return codec
    return "i8"


def spine_targets(res_dir=RES):
    """[(name, (rows, cols))] — the same tensors convert_spine_int8.py converts."""
    out = []
    for name, t in INV.items():
        if t["dtype"] != "BF16" or len(t["shape"]) != 2:
            continue
        if any(x in name for x in EXCLUDE):
            continue
        if not os.path.exists(os.path.join(res_dir, name)):
            continue
        out.append((name, tuple(t["shape"])))
    out.sort()
    return out


def n_groups(cols, g):
    return (cols + g - 1) // g


def blob_sizes(rows, cols, g):
    """(bytes of .i4, bytes of .s4) for a [rows, cols] tensor at group size g."""
    ng = n_groups(cols, g)
    return rows * (ng * g) // 2, rows * ng * 2


def quantize_block(w, g):
    """w: fp32 [nr, cw] with cw % g == 0 -> (packed uint8 [nr, cw/2],
    scales fp16 [nr, ng], dequantized fp32 [nr, cw]).  The returned dequant is
    bit-identical to what int4_loader.py reconstructs."""
    nr, cw = w.shape
    ng = cw // g
    wg = w.reshape(nr, ng, g)
    s16 = (np.abs(wg).max(axis=2) / 7.0).astype(np.float16)
    s16[s16 == 0] = np.float16(6e-8)                 # all-zero group guard
    s = s16.astype(np.float32)[:, :, None]           # quantize with the STORED value
    q = np.clip(np.rint(wg / s), -7, 7).astype(np.int8)
    deq = (q.astype(np.float32) * s).reshape(nr, cw)
    nib = (q.reshape(nr, cw).view(np.uint8) & np.uint8(0x0F))
    packed = (nib[:, 0::2] | (nib[:, 1::2] << np.uint8(4))).astype(np.uint8)
    return packed, s16, deq


def convert_one(name, rows, cols, g, out_dir, res_dir=RES, chunk_elems=8 << 20,
                measure=False):
    """Stream one tensor bf16 -> .i4/.s4. Returns (rel_fro_err, max_abs_err) or
    (None, None) when measure=False."""
    op = os.path.join(out_dir, name + ".i4")
    sp = os.path.join(out_dir, name + ".s4")
    cw = n_groups(cols, g) * g                       # column count after zero-pad
    src = os.path.join(res_dir, name)
    rows_per = max(1, min(rows, chunk_elems // max(cols, 1)))
    se, sw, mx = 0.0, 0.0, 0.0
    with open(op + ".tmp", "wb") as fq, open(sp + ".tmp", "wb") as fs:
        for r0 in range(0, rows, rows_per):
            nr = min(rows_per, rows - r0)
            raw = np.fromfile(src, dtype=np.uint16, count=nr * cols,
                              offset=r0 * cols * 2).reshape(nr, cols)
            w = (raw.astype(np.uint32) << 16).view(np.float32)
            if cw != cols:                            # pad the tail group with zeros
                w = np.concatenate([w, np.zeros((nr, cw - cols), np.float32)], axis=1)
            packed, s16, deq = quantize_block(w, g)
            packed.tofile(fq)
            s16.tofile(fs)
            if measure:
                e = deq[:, :cols] - w[:, :cols]
                se += float(np.dot(e.ravel(), e.ravel()))
                sw += float(np.dot(w[:, :cols].ravel(), w[:, :cols].ravel()))
                mx = max(mx, float(np.abs(e).max()))
            del raw, w, packed, s16, deq
    os.replace(op + ".tmp", op)
    os.replace(sp + ".tmp", sp)
    if not measure:
        return None, None
    return (se / (sw + 1e-30)) ** 0.5, mx


def convert_one_codec(name, rows, cols, codec, g, out_dir, res_dir=RES,
                      chunk_elems=8 << 20, measure=True):
    """Stream one bf16 tensor -> the codec's two blobs. Returns (rel_fro, max_abs)."""
    pe, se = spine_codec.EXT[codec]
    op, sp = os.path.join(out_dir, name + pe), os.path.join(out_dir, name + se)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    src = os.path.join(res_dir, name)
    rows_per = max(1, min(rows, chunk_elems // max(cols, 1)))
    if codec != "i8":                       # keep whole groups inside a chunk
        rows_per = max(1, rows_per)
    sq, sw, mx = 0.0, 0.0, 0.0
    with open(op + ".tmp", "wb") as fq, open(sp + ".tmp", "wb") as fs:
        for r0 in range(0, rows, rows_per):
            nr = min(rows_per, rows - r0)
            raw = np.fromfile(src, dtype=np.uint16, count=nr * cols,
                              offset=r0 * cols * 2).reshape(nr, cols)
            w = (raw.astype(np.uint32) << 16).view(np.float32)
            packed, s16, deq = spine_codec.quantize(w, codec, g)
            packed.tofile(fq)
            s16.tofile(fs)
            if measure:
                e = deq - w
                sq += float(np.dot(e.ravel(), e.ravel()))
                sw += float(np.dot(w.ravel(), w.ravel()))
                mx = max(mx, float(np.abs(e).max()))
            del raw, w, packed, s16, deq
    os.replace(op + ".tmp", op)
    os.replace(sp + ".tmp", sp)
    return ((sq / (sw + 1e-30)) ** 0.5, mx) if measure else (None, None)


def parse_layers(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def run_policy(args):
    """Mixed-precision conversion. Writes only the tensors the policy moves off
    i8; a loader falls back to the int8 spine for the rest."""
    rules = parse_policy(args.policy or args.policy_roles)
    g = args.group
    out_dir = args.out
    layers = parse_layers(args.layers)
    targets = spine_targets()
    # embed_tokens is never read by the per-token spine walk (kimi_run.LazyEmbed
    # reads single bf16 rows straight out of k3-resident), so quantizing it costs
    # disk and buys nothing. Policy mode always skips it; --keep-embed overrides.
    if not args.keep_embed:
        targets = [t for t in targets if not t[0].endswith("embed_tokens.weight")]

    plan, i8_keep = [], []
    for name, (rows, cols) in targets:
        codec = codec_for(name, rules)
        if layers is not None and layer_of(name) not in layers and layer_of(name) >= 0:
            codec = "i8"
        (i8_keep if codec == "i8" else plan).append((name, rows, cols, codec))

    def gb(items, force=None):
        return sum(sum(spine_codec.blob_sizes(force or c, r, cl, g))
                   for _, r, cl, c in items) / 1e9

    base_i8 = gb(plan, "i8") + gb(i8_keep, "i8")
    now = gb(plan) + gb(i8_keep, "i8")
    per_codec = {}
    for _, r, c, cd in plan:
        per_codec[cd] = per_codec.get(cd, 0) + sum(spine_codec.blob_sizes(cd, r, c, g))
    print(f"[mixed] policy={args.policy or args.policy_roles} g={g}: "
          f"{len(plan)} tensors move off int8 "
          f"({', '.join(f'{k} {v/1e9:.1f} GB' for k, v in sorted(per_codec.items()))}), "
          f"{len(i8_keep)} stay int8", flush=True)
    print(f"[mixed] whole spine {base_i8:.1f} GB int8 -> {now:.1f} GB "
          f"({100*(1-now/base_i8):.1f}% smaller); per-token read drops by "
          f"{base_i8-now:.1f} GB", flush=True)
    if args.dry_run:
        by_role = {}
        for name, r, c, cd in plan:
            k = (role_of(name), cd)
            by_role[k] = by_role.get(k, 0) + sum(spine_codec.blob_sizes(cd, r, c, g))
        for (role, cd), b in sorted(by_role.items(), key=lambda kv: -kv[1]):
            print(f"   {cd}  {role:55s} {b/1e9:6.2f} GB")
        return

    os.makedirs(out_dir, exist_ok=True)
    meta = {"format": "mixed-spine", "version": 1, "group": g,
            "codecs": {c: spine_codec.bytes_per_elt(c, 7168, g) for c in spine_codec.CODECS},
            "policy": args.policy or args.policy_roles, "rules": rules,
            "layers": sorted(layers) if layers else "all",
            "int8_fallback_dir": os.path.relpath(INT8, K3),
            "n_converted": len(plan), "n_int8": len(i8_keep),
            "spine_gb_int8": round(base_i8, 2), "spine_gb_mixed": round(now, 2)}
    json.dump(meta, open(os.path.join(os.path.dirname(out_dir.rstrip("/")),
                                      "meta.json"), "w"), indent=1)

    t0, done, errs = time.time(), 0, []
    total = sum(r * c * 2 for _, r, c, _ in plan)
    for i, (name, rows, cols, codec) in enumerate(plan):
        pe, se = spine_codec.EXT[codec]
        eb, sb = spine_codec.blob_sizes(codec, rows, cols, g)
        op, sp = os.path.join(out_dir, name + pe), os.path.join(out_dir, name + se)
        if (not args.force and os.path.exists(op) and os.path.getsize(op) == eb
                and os.path.exists(sp) and os.path.getsize(sp) == sb):
            done += rows * cols * 2
            continue
        rel, mx = convert_one_codec(name, rows, cols, codec, g, out_dir,
                                    measure=args.verify)
        done += rows * cols * 2
        if rel is not None:
            errs.append((name, codec, rel))
            print(f"  {codec} {name:66s} [{rows},{cols}] rel_fro={rel:.4e} "
                  f"max_abs={mx:.2e}", flush=True)
        elif i % 50 == 0:
            el = max(time.time() - t0, 1e-9)
            print(f"{i}/{len(plan)} {done/1e9:.1f}/{total/1e9:.1f} GB bf16 read "
                  f"({done/1e9/el:.2f} GB/s, eta {(total-done)/1e9/max(done/1e9/el,0.01)/60:.0f} min)",
                  flush=True)
    if args.int8_mode != "skip":
        n = 0
        for name, rows, cols, _ in i8_keep:
            for ext in (".i8", ".sc"):
                src, dst = os.path.join(INT8, name + ext), os.path.join(out_dir, name + ext)
                if os.path.exists(dst) or not os.path.exists(src):
                    continue
                (os.link if args.int8_mode == "link" else _copy)(src, dst)
                n += 1
        print(f"[mixed] {args.int8_mode}ed {n} int8 blobs into {out_dir}", flush=True)
    if errs:
        for cd in sorted({c for _, c, _ in errs}):
            e = np.array([x[2] for x in errs if x[1] == cd])
            print(f"[mixed] {cd}: rel_fro min {e.min():.4e} / median "
                  f"{np.median(e):.4e} / max {e.max():.4e} over {e.size} tensors")
    print(f"DONE {done/1e9:.1f} GB bf16 read in {(time.time()-t0)/60:.1f} min "
          f"-> {out_dir}", flush=True)


def _copy(src, dst):
    import shutil
    shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group", "-g", type=int, default=64,
                    help="elements per scale along a row (default 64)")
    ap.add_argument("--out", default=os.path.join(K3, "k3-resident-int4/tensors"))
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="convert only N tensors, evenly spread over the spine, "
                         "and report per-tensor error (cheap validation)")
    ap.add_argument("--names", nargs="+", default=None,
                    help="convert only these exact tensor names (implies --verify)")
    ap.add_argument("--skip-larger-than", type=float, default=0.0, metavar="GB",
                    help="skip tensors whose bf16 blob exceeds this (0 = no limit)")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip embed_tokens. kimi_run.LazyEmbed reads single bf16 rows "
                         "straight out of k3-resident, so the quantized copy of "
                         "embed_tokens (2.35 GB bf16) is never read — converting it "
                         "costs disk and buys no per-token I/O. Off by default only to "
                         "keep the tensor set identical to convert_spine_int8.py.")
    ap.add_argument("--verify", action="store_true",
                    help="measure reconstruction error on every tensor (slower)")
    ap.add_argument("--force", action="store_true", help="reconvert existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="list targets and sizes only")
    # ---- mixed-precision mode ----
    ap.add_argument("--policy", default=None, metavar="NAME|SPEC",
                    help="mixed-precision mode: a built-in policy name "
                         f"({'/'.join(POLICIES)}), a 'role_substr=codec,...' spec, "
                         "or @file.json. Codecs: i4 (0.53 B/elt), i6 (0.78), "
                         "i8 (1.0, left to the int8 spine).")
    ap.add_argument("--policy-roles", default=None, metavar="SPEC",
                    help="alias for --policy with an inline spec")
    ap.add_argument("--layers", default=None, metavar="A-B,C",
                    help="restrict the policy to these layer indices; every "
                         "other layer stays int8 (used to prove the path "
                         "end-to-end without converting the whole spine)")
    ap.add_argument("--keep-embed", action="store_true",
                    help="policy mode: also quantize embed_tokens (default skip "
                         "— it is never read per token)")
    ap.add_argument("--int8-mode", choices=("skip", "link", "copy"), default="skip",
                    help="what to do with the tensors the policy leaves at int8: "
                         "skip (default — the loader falls back to "
                         "k3-resident-int8), link (hardlink into the mixed dir), "
                         "or copy (self-contained, +53 GB)")
    args = ap.parse_args()
    g = args.group
    if g < 8 or g % 8 or g > 1024:
        sys.exit(f"--group must be a multiple of 8 in [8,1024], got {g}")
    if args.policy or args.policy_roles:
        if args.out == os.path.join(K3, "k3-resident-int4/tensors"):
            args.out = os.path.join(K3, "k3-resident-mixed/tensors")
        if g % 4:
            sys.exit("--group must be a multiple of 4 for the i6 packing")
        return run_policy(args)
    os.makedirs(args.out, exist_ok=True)

    targets = spine_targets()
    if args.no_embed:
        targets = [t for t in targets if not t[0].endswith("embed_tokens.weight")]
    if args.skip_larger_than:
        lim = args.skip_larger_than * 1e9
        targets = [t for t in targets if t[1][0] * t[1][1] * 2 <= lim]
    if args.names:
        want = set(args.names)
        targets = [t for t in targets if t[0] in want]
        missing = want - {t[0] for t in targets}
        if missing:
            sys.exit(f"not convertible / not present: {sorted(missing)}")
        args.verify = True
    elif args.sample:
        n = min(args.sample, len(targets))
        step = max(1, len(targets) // n)
        targets = targets[::step][:n]
        args.verify = True

    bf16_b = sum(r * c * 2 for _, (r, c) in targets)
    i4_b = sum(sum(blob_sizes(r, c, g)) for _, (r, c) in targets)
    print(f"[int4] {len(targets)} tensors, g={g}: {bf16_b/1e9:.1f} GB bf16 -> "
          f"{i4_b/1e9:.1f} GB int4+scales (x{bf16_b/i4_b:.2f}); out={args.out}",
          flush=True)
    if args.dry_run:
        for name, (r, c) in targets[:20]:
            print(f"   {name}  [{r},{c}]  {sum(blob_sizes(r,c,g))/1e6:.1f} MB")
        return

    meta = {"format": FORMAT, "version": VERSION, "group": g,
            "packing": "uint8[rows, ceil(cols/g)*g/2], low nibble = even column",
            "scale": "float16[rows, ceil(cols/g)], quantized with the stored fp16 value",
            "levels": "signed 4-bit, clipped to [-7,7] (-8 unused)",
            "excluded": list(EXCLUDE), "n_tensors": len(targets),
            "partial": bool(args.sample or args.names or args.skip_larger_than)}
    mp = os.path.join(os.path.dirname(args.out.rstrip("/")), "meta.json")
    prev = json.load(open(mp)) if os.path.exists(mp) else None
    if prev and prev.get("group") != g and not args.force:
        sys.exit(f"{mp} says group={prev['group']} but --group={g}; the directory "
                 f"would be a mix. Use a different --out or pass --force.")
    json.dump(meta, open(mp, "w"), indent=1)

    t0, done, errs = time.time(), 0, []
    for i, (name, (rows, cols)) in enumerate(targets):
        eb, sb = blob_sizes(rows, cols, g)
        op, sp = os.path.join(args.out, name + ".i4"), os.path.join(args.out, name + ".s4")
        if (not args.force and os.path.exists(op) and os.path.getsize(op) == eb
                and os.path.exists(sp) and os.path.getsize(sp) == sb):
            done += rows * cols * 2
            continue
        rel, mx = convert_one(name, rows, cols, g, args.out,
                              measure=(args.verify or args.sample > 0))
        done += rows * cols * 2
        if rel is not None:
            errs.append((name, (rows, cols), rel, mx))
            print(f"  {name:70s} [{rows},{cols}] rel_fro={rel:.3e} max_abs={mx:.2e}",
                  flush=True)
        if not args.verify and i % 100 == 0:
            el = max(time.time() - t0, 1e-9)
            rate = done / 1e9 / el
            print(f"{i}/{len(targets)} {done/1e9:.1f}/{bf16_b/1e9:.1f} GB "
                  f"({rate:.2f} GB/s, eta {(bf16_b-done)/1e9/max(rate,0.01)/60:.0f} min)",
                  flush=True)
    if errs:
        w = np.array([r * c for _, (r, c), _, _ in errs], dtype=np.float64)
        e = np.array([x[2] for x in errs])
        print(f"\n[int4 g={g}] rel_fro over {len(errs)} tensors: "
              f"min {e.min():.3e} / median {np.median(e):.3e} / max {e.max():.3e} "
              f"/ size-weighted {float((e*w).sum()/w.sum()):.3e}")
    print(f"DONE {done/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
