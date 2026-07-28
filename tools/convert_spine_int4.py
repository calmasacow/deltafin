#!/usr/bin/env python3
"""Full-spine int4 conversion (OPT-IN, QUALITY-GATED — never a default).

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
import argparse, json, os, sys, time
import numpy as np

K3 = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(K3, "k3-resident/tensors")
INV = json.load(open(os.path.join(K3, "k3-meta/tensor_inventory_offsets.json")))

# Identical selection rule to convert_spine_int8.py — keep these in sync.
EXCLUDE = ("conv1d", "norm", "A_log", "dt_bias", "res_proj",
           "e_score_correction_bias", ".experts.", "vision", "mm_projector")

FORMAT = "int4-sym-group"
VERSION = 1


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
    args = ap.parse_args()
    g = args.group
    if g < 8 or g % 8 or g > 1024:
        sys.exit(f"--group must be a multiple of 8 in [8,1024], got {g}")
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
