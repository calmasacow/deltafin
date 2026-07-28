#!/usr/bin/env python3
"""One-time setup for a fresh Deltafin clone.

Always downloads the small stuff from huggingface.co/moonshotai/Kimi-K3:
  k3-meta/      config, tokenizer, and Moonshot's modeling files (a few MB)
  tools/k3pkg/  import home for those modeling modules
plus k3-meta/tensor_inventory_offsets.json, built by reading only the 96 shard
HEADERS via range requests. Idempotent; safe to re-run.

Then it installs the weights, in one of two modes:

  --full    (default when the disk allows it, and STRONGLY recommended)
            resident spine (~114 GB) + every routed expert (~1.45 TB).
            Nothing is fetched over the network at inference time, so every
            prompt runs at full speed.

  --stream  resident spine only (~114 GB). Experts are fetched over HTTP as the
            router asks for them. This works, but a token needs 25.8 GB of
            expert data: ~4 s from local NVMe versus MINUTES over the network.
            Only text whose experts happen to be cached runs at full speed, so
            in practice most prompts are several times slower.

With no flag, setup picks --full if there is room and falls back to --stream
with a warning if there isn't.
"""
import concurrent.futures
import json
import os
import shutil
import struct
import sys
import urllib.request

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "k3-meta")
PKG = os.path.join(ROOT, "tools", "k3pkg")
BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/"

FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json",
    "tokenization_kimi.py", "encoding_k3.py", "tiktoken.model",
    "modeling_kimi_linear.py", "modeling_kimi_k3.py", "configuration_kimi_k3.py",
]
PKG_FILES = ["modeling_kimi_linear.py", "modeling_kimi_k3.py", "configuration_kimi_k3.py"]


def fetch(name):
    dst = os.path.join(META, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return f"  {name}: already present"
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "deltafin-setup"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(dst + ".part", dst)
    return f"  {name}: downloaded"


def shard_header(i):
    shard = f"model-{i:05d}-of-000096.safetensors"

    def rng(start, end):
        req = urllib.request.Request(
            BASE + shard, headers={"Range": f"bytes={start}-{end}",
                                   "User-Agent": "deltafin-setup"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    n = struct.unpack("<Q", rng(0, 7))[0]
    h = json.loads(rng(8, 8 + n - 1))
    h.pop("__metadata__", None)
    return shard, n, h


SPINE_BYTES = 114e9
EXPERTS_BYTES = 82432 * 17547264      # ~1.45 TB
HEADROOM = 100e9


def _remaining_bytes():
    """What still has to be downloaded, so re-running as an upgrade doesn't
    demand space for weights that are already on disk."""
    ecache = os.path.join(ROOT, "k3-experts")
    try:
        have = sum(1 for f in os.listdir(ecache) if f.endswith((".bin", ".npz")))
    except FileNotFoundError:
        have = 0
    experts_left = max(0, 82432 - have) * 17547264
    spine = os.path.join(ROOT, "k3-resident", "tensors")
    try:
        spine_have = sum(os.path.getsize(os.path.join(spine, f))
                         for f in os.listdir(spine))
    except FileNotFoundError:
        spine_have = 0
    return max(0, SPINE_BYTES - spine_have), experts_left


def _exclude_from_spotlight():
    """macOS indexes new files aggressively. 1.5 TB of weight blobs sends
    corespotlightd to ~50% CPU for hours, which contends with inference and
    makes any benchmark meaningless. A .metadata_never_index marker stops it."""
    for d in ("k3-experts", "k3-resident", "k3-resident-int8", "k3-resident-int4"):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            open(os.path.join(p, ".metadata_never_index"), "a").close()


def install_weights(mode):
    import shutil as _sh
    import subprocess
    free = _sh.disk_usage(ROOT).free
    spine_left, experts_left = _remaining_bytes()
    need_full = spine_left + experts_left + HEADROOM
    if mode is None:
        mode = "full" if free >= need_full else "stream"
        if mode == "stream":
            print("\n" + "!" * 72)
            print(f"  Not enough disk for the full install: it needs "
                  f"{need_full/1e12:.2f} TB more (weights + 100 GB headroom)")
            print(f"  and this volume has {free/1e12:.2f} TB free — about "
                  f"{(need_full-free)/1e12:.2f} TB short.")
            print("  Falling back to STREAMING, which is SUBSTANTIALLY SLOWER:")
            print("  every prompt whose experts aren't cached fetches them over")
            print("  the network — minutes per token instead of about one.")
            print(f"  Freeing ~{(need_full-free)/1e12:.2f} TB and re-running with")
            print("  --full is the single biggest speedup available.")
            print("!" * 72 + "\n")
    if mode == "full" and free < need_full:
        print(f"--full still needs {need_full/1e12:.2f} TB (of which "
              f"{experts_left/1e12:.2f} TB is experts), but only {free/1e12:.2f} TB is free.")
        print("Free up space, or use --stream for now and run")
        print("tools/fetch_experts_all.py later.")
        sys.exit(1)

    py = sys.executable
    os.makedirs(os.path.join(ROOT, "k3-experts"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "k3-resident"), exist_ok=True)
    _exclude_from_spotlight()
    print(f"\n== installing weights: {mode} ==", flush=True)
    subprocess.check_call([py, os.path.join(ROOT, "tools", "fetch_spine.py")])
    if mode == "full":
        subprocess.check_call([py, os.path.join(ROOT, "tools", "fetch_experts_all.py")])
        print("\nFull install complete — no network needed at inference time.")
    else:
        print("\nStreaming install complete (spine only).")
        print("Reminder: experts are fetched over the network on demand, so most")
        print("prompts will be several times slower than a full install. When you")
        print("have ~1.45 TB free:  python tools/fetch_experts_all.py")
    print("Optional next step (halves per-token I/O):")
    print("  python tools/convert_spine_int8.py")


def main():
    ap_mode = None
    if "--full" in sys.argv:
        ap_mode = "full"
    elif "--stream" in sys.argv:
        ap_mode = "stream"
    meta_only = "--meta-only" in sys.argv

    os.makedirs(META, exist_ok=True)
    print(f"Deltafin setup -> {META}")
    for name in FILES:
        print(fetch(name))
    for name in PKG_FILES:
        shutil.copy2(os.path.join(META, name), os.path.join(PKG, name))
    print(f"  modeling files copied into tools/k3pkg/")

    inv_path = os.path.join(META, "tensor_inventory_offsets.json")
    if os.path.exists(inv_path):
        print("  tensor inventory: already present")
        if meta_only:
            return
        install_weights(ap_mode)
        return
    print("  building tensor inventory from 96 shard headers (range requests)...")
    inv = {}
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        for shard, hlen, h in ex.map(shard_header, range(1, 97)):
            for name, info in h.items():
                inv[name] = {"dtype": info["dtype"], "shape": info["shape"],
                             "offsets": info["data_offsets"], "shard": shard,
                             "hlen": hlen}
            sys.stderr.write(".")
    sys.stderr.write("\n")
    with open(inv_path + ".part", "w") as f:
        json.dump(inv, f)
    os.replace(inv_path + ".part", inv_path)
    print(f"  tensor inventory: {len(inv)} tensors indexed")
    if meta_only:
        print("Done (--meta-only). Weights not installed.")
        return
    install_weights(ap_mode)


if __name__ == "__main__":
    main()
