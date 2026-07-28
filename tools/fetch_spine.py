#!/usr/bin/env python3
"""Download Kimi-K3's resident (non-routed-expert) tensors via HTTP Range requests.
Writes one raw .bin per tensor under k3-resident/tensors/ (name = tensor name).
Resumable: existing files with the right size are skipped."""
import json, os, sys, time, urllib.request, concurrent.futures, threading

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV = json.load(open(os.path.join(ROOT, "k3-meta/tensor_inventory_offsets.json")))
OUT = os.path.join(ROOT, "k3-resident/tensors")
os.makedirs(OUT, exist_ok=True)

work = []
for name, t in INV.items():
    if ".experts." in name:
        continue
    size = t["offsets"][1] - t["offsets"][0]
    path = os.path.join(OUT, name)
    if os.path.exists(path) and os.path.getsize(path) == size:
        continue
    work.append((name, t, size, path))
work.sort(key=lambda w: (w[1]["shard"], w[1]["offsets"][0]))
total = sum(w[2] for w in work)
print(f"to fetch: {len(work)} tensors, {total/1e9:.1f} GB", flush=True)

done_b = 0
lock = threading.Lock()
t0 = time.time()

def fetch(item):
    global done_b
    name, t, size, path = item
    start = 8 + t["hlen"] + t["offsets"][0]
    req = urllib.request.Request(BASE + t["shard"],
                                 headers={"Range": f"bytes={start}-{start+size-1}"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(path + ".part") != size:
                raise IOError("short read")
            os.replace(path + ".part", path)
            with lock:
                done_b += size
            return
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    print(f"FAILED {name}: giving up", flush=True)

with concurrent.futures.ThreadPoolExecutor(10) as ex:
    futs = [ex.submit(fetch, w) for w in work]
    last = 0
    for i, f in enumerate(concurrent.futures.as_completed(futs)):
        f.result()
        now = time.time()
        if now - last > 30:
            last = now
            mbs = done_b / 1e6 / max(now - t0, 1)
            print(f"progress {done_b/1e9:.1f}/{total/1e9:.1f} GB  {mbs:.0f} MB/s  eta {(total-done_b)/1e6/max(mbs,1)/60:.0f} min", flush=True)
print(f"DONE {done_b/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min", flush=True)
