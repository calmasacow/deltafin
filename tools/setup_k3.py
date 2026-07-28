#!/usr/bin/env python3
"""One-time setup for a fresh Deltafin clone.

Downloads from huggingface.co/moonshotai/Kimi-K3 (a few MB + the tokenizer):
  k3-meta/      config.json, generation_config.json, tokenizer_config.json,
                tokenization_kimi.py, encoding_k3.py, tiktoken.model,
                modeling_kimi_linear.py, modeling_kimi_k3.py, configuration_kimi_k3.py
  tools/k3pkg/  copies of the three modeling/configuration modules (import home)

Then builds k3-meta/tensor_inventory_offsets.json by reading only the 96 shard
HEADERS via HTTP range requests (~no download cost) — the index every other tool
uses to locate tensors. Idempotent; safe to re-run.

The model weights themselves are NOT downloaded here: run tools/fetch_spine.py
for the resident spine (~114 GB); routed experts stream on demand at inference.
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


def main():
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
    print("Done. Next: tools/fetch_spine.py (~114 GB), then tools/kimi_run.py")


if __name__ == "__main__":
    main()
