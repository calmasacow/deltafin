"""fetch_v2: coalesced, persistent-connection expert fetcher for lazy-K3.

Drop-in replacement for k3loader's expert fetch path:
    fetch_expert_raw(layer, eid) -> {"w1": (packed u8, scale u8), "w2": ..., "w3": ...}
    fetch_experts(layer, eids, workers=4, dequant=...)  (same signature/semantics)

Differences vs baseline (k3loader._range_fetch):
  * ONE Range request per expert (17.55 MB contiguous span) instead of 6.
  * The HF resolve -> CDN 302 redirect is resolved ONCE per shard and cached
    (signed URL, ~1h validity; auto re-resolve on 403/expiry).
  * Persistent keep-alive connections to the CDN host, small pool (default 4).
  * Optional multi-expert span coalescing in fetch_experts() when selected
    experts are file-adjacent (lexicographic eid order, zero-gap layout).
  * Optional httpx HTTP/2 backend (BACKEND="httpx") for benchmarking.

Expert layout facts (verified against tensor_inventory_offsets.json for all
82432 experts): each expert's 6 tensors are contiguous in-shard in order
w1_p, w1_s, w2_p, w2_s, w3_p, w3_s with fixed sizes; each MoE layer's 896
experts occupy ONE shard, back-to-back with zero gaps, sorted by str(eid).
"""
import http.client
import json
import os
import re
import ssl
import threading
import time
import urllib.parse
import concurrent.futures

import numpy as np

ROOT = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV_PATH = os.path.join(ROOT, "k3-meta/tensor_inventory_offsets.json")
BASE_HOST = os.environ.get("K3_HF_HOST", "huggingface.co")
BASE_PATH = os.environ.get("K3_HF_PATH", "/moonshotai/Kimi-K3/resolve/main/")
ECACHE = os.path.join(ROOT, "k3-experts")
os.makedirs(ECACHE, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
IDX_PATH = os.path.join(_HERE, "expert_index.npy")
IDX_META_PATH = os.path.join(_HERE, "expert_index.meta.json")

# ---- fixed intra-expert layout (offset, nbytes, shape) --------------------
EXPERT_SPAN = 17547264
_P, _S = 5505024, 344064
LAYOUT = [
    ("w1_p", 0,            _P, (3072, 1792)),
    ("w1_s", _P,           _S, (3072, 112)),
    ("w2_p", _P + _S,      _P, (3584, 1536)),
    ("w2_s", 2 * _P + _S,  _S, (3584, 96)),
    ("w3_p", 2 * (_P + _S), _P, (3072, 1792)),
    ("w3_s", 3 * _P + 2 * _S, _S, (3072, 112)),
]

USER_AGENT = "k3-lazy-fetch/2.0"
MAX_CONNS = int(os.environ.get("K3_FETCH_CONNS", "4"))
BACKEND = os.environ.get("K3_FETCH_BACKEND", "httpclient")  # or "httpx"
MAX_COALESCE = int(os.environ.get("K3_FETCH_COALESCE", "4"))  # experts per merged range

stats = {
    "expert_http": 0, "expert_disk": 0, "http_bytes": 0, "http_s": 0.0,
    "requests": 0, "new_conns": 0, "conn_s": 0.0, "resolves": 0,
    "resolve_s": 0.0, "retries": 0, "coalesced_spans": 0,
}
_stats_lock = threading.Lock()


def _bump(**kw):
    with _stats_lock:
        for k, v in kw.items():
            stats[k] += v


# ---------------------------------------------------------------------------
# expert index: (layer, eid) -> (shard name, absolute byte start)
# ---------------------------------------------------------------------------
_index = None          # dict (L, E) -> (shard_num, start)
_shard_template = None
_index_lock = threading.Lock()


def _build_index():
    inv = json.load(open(INV_PATH))
    pat = re.compile(r"language_model\.model\.layers\.(\d+)\.block_sparse_moe"
                     r"\.experts\.(\d+)\.w1\.weight_packed$")
    rows, template = [], None
    for name, t in inv.items():
        m = pat.match(name)
        if not m:
            continue
        L, E = int(m.group(1)), int(m.group(2))
        start = 8 + t["hlen"] + t["offsets"][0]
        snum = int(t["shard"].split("-")[1])
        rows.append((L, E, snum, start, start + EXPERT_SPAN))
        if template is None:
            template = re.sub(r"(?<=model-)\d+", "{:05d}", t["shard"], count=1)
    arr = np.array(sorted(rows), dtype=np.int64)
    tmp = IDX_PATH + f".tmp{os.getpid()}"
    np.save(tmp, arr)
    os.replace(tmp + ".npy" if os.path.exists(tmp + ".npy") else tmp, IDX_PATH)
    json.dump({"shard_template": template}, open(IDX_META_PATH, "w"))
    return arr, template


def _get_index():
    global _index, _shard_template
    if _index is not None:
        return _index
    with _index_lock:
        if _index is None:
            if os.path.exists(IDX_PATH) and os.path.exists(IDX_META_PATH):
                arr = np.load(IDX_PATH)
                _shard_template = json.load(open(IDX_META_PATH))["shard_template"]
            else:
                arr, _shard_template = _build_index()
            _index = {(int(r[0]), int(r[1])): (int(r[2]), int(r[3])) for r in arr}
    return _index


def expert_span(layer, eid):
    """(shard_name, abs_start, abs_end) for one expert's contiguous 6-tensor span."""
    snum, start = _get_index()[(layer, eid)]
    return _shard_template.format(snum), start, start + EXPERT_SPAN


# ---------------------------------------------------------------------------
# redirect resolver: shard -> signed CDN URL (cached, thread-safe)
# ---------------------------------------------------------------------------
_ssl_ctx = ssl.create_default_context()
_resolved = {}          # shard -> (host, path_with_query, expires_epoch)
_resolve_lock = threading.Lock()


def _resolve(shard, force=False):
    now = time.time()
    with _resolve_lock:
        ent = _resolved.get(shard)
        if ent and not force and ent[2] - now > 300:
            return ent
    t0 = time.time()
    c = http.client.HTTPSConnection(BASE_HOST, timeout=30, context=_ssl_ctx)
    try:
        c.request("HEAD", BASE_PATH + shard,
                  headers={"User-Agent": USER_AGENT})
        r = c.getresponse()
        r.read()
        if r.status not in (301, 302, 303, 307, 308):
            raise IOError(f"resolve {shard}: expected redirect, got {r.status}")
        loc = r.getheader("Location")
    finally:
        c.close()
    u = urllib.parse.urlsplit(loc)
    q = urllib.parse.parse_qs(u.query)
    exp = int(q.get("Expires", [now + 2700])[0])
    path = u.path + ("?" + u.query if u.query else "")
    ent = (u.netloc, path, exp)
    with _resolve_lock:
        _resolved[shard] = ent
    _bump(resolves=1, resolve_s=time.time() - t0)
    return ent


# ---------------------------------------------------------------------------
# persistent connection pool (http.client backend)
# ---------------------------------------------------------------------------
class _Pool:
    def __init__(self, max_conns=MAX_CONNS):
        self._sem = threading.BoundedSemaphore(max_conns)
        self._idle = []          # [(host, conn)]
        self._lock = threading.Lock()

    def _get(self, host):
        self._sem.acquire()
        with self._lock:
            for i, (h, c) in enumerate(self._idle):
                if h == host:
                    self._idle.pop(i)
                    return c
            # evict one idle conn to a stale host if pool is saturated with them
            if self._idle:
                _, c = self._idle.pop(0)
                c.close()
        t0 = time.time()
        c = http.client.HTTPSConnection(host, timeout=120, context=_ssl_ctx)
        c.connect()
        _bump(new_conns=1, conn_s=time.time() - t0)
        return c

    def _put(self, host, conn, reusable):
        with self._lock:
            if reusable:
                self._idle.append((host, conn))
        if not reusable:
            try:
                conn.close()
            except Exception:
                pass
        self._sem.release()

    def range_get(self, shard, start, size, retries=6):
        """Fetch [start, start+size) of shard via persistent conn; returns bytes."""
        last = None
        for attempt in range(retries):
            host, path, _ = _resolve(shard, force=attempt >= 2)
            conn = self._get(host)
            reusable = False
            try:
                t0 = time.time()
                conn.request("GET", path, headers={
                    "Range": f"bytes={start}-{start + size - 1}",
                    "User-Agent": USER_AGENT,
                })
                r = conn.getresponse()
                if r.status == 403:          # signed URL expired
                    r.read()
                    _resolve(shard, force=True)
                    raise IOError("403 signed-url expired")
                if r.status != 206:
                    r.read()
                    raise IOError(f"status {r.status}")
                buf = r.read(size)
                if len(buf) != size:
                    raise IOError(f"short read {len(buf)}/{size}")
                reusable = not r.will_close
                _bump(requests=1, http_bytes=size, http_s=time.time() - t0)
                return buf
            except Exception as e:
                last = e
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                _bump(retries=1)
                if attempt == retries - 1:
                    raise
                time.sleep(min(1.5 * attempt, 6.0))
            finally:
                if conn is not None:
                    self._put(host, conn, reusable)
                else:
                    self._sem.release()
        raise last


_pool = _Pool()

# ---------------------------------------------------------------------------
# optional httpx HTTP/2 backend
# ---------------------------------------------------------------------------
_httpx_client = None
_httpx_lock = threading.Lock()


def _get_httpx():
    global _httpx_client
    if _httpx_client is None:
        with _httpx_lock:
            if _httpx_client is None:
                import httpx
                _httpx_client = httpx.Client(
                    http2=True, timeout=120.0,
                    limits=httpx.Limits(max_connections=MAX_CONNS,
                                        max_keepalive_connections=MAX_CONNS),
                    headers={"User-Agent": USER_AGENT})
    return _httpx_client


def _httpx_range_get(shard, start, size, retries=6):
    last = None
    for attempt in range(retries):
        host, path, _ = _resolve(shard, force=attempt >= 2)
        try:
            t0 = time.time()
            r = _get_httpx().get(
                f"https://{host}{path}",
                headers={"Range": f"bytes={start}-{start + size - 1}"})
            if r.status_code == 403:
                _resolve(shard, force=True)
                raise IOError("403 signed-url expired")
            if r.status_code != 206:
                raise IOError(f"status {r.status_code}")
            buf = r.content
            if len(buf) != size:
                raise IOError(f"short read {len(buf)}/{size}")
            _bump(requests=1, http_bytes=size, http_s=time.time() - t0)
            return buf
        except Exception as e:
            last = e
            _bump(retries=1)
            if attempt == retries - 1:
                raise
            time.sleep(min(1.5 * attempt, 6.0))
    raise last


def _range_get(shard, start, size):
    if BACKEND == "httpx":
        return _httpx_range_get(shard, start, size)
    return _pool.range_get(shard, start, size)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _slice_expert(buf, base=0):
    """Slice one expert's 6 tensors out of a fetched buffer."""
    out = {}
    for name, off, nb, shape in LAYOUT:
        w, kind = name.split("_")
        a = np.frombuffer(buf, dtype=np.uint8, count=nb,
                          offset=base + off).reshape(shape)
        out.setdefault(w, {})[kind] = a
    return {w: (d["p"], d["s"]) for w, d in out.items()}


def _cache_path(layer, eid):
    return os.path.join(ECACHE, f"L{layer}-E{eid}.npz")


def _cache_path_bin(layer, eid):
    # raw format: the expert's 17,547,264-byte shard span verbatim (w1_p w1_s w2_p
    # w2_s w3_p w3_s) — zero-parse, mmap-able; npz kept as read fallback
    return os.path.join(ECACHE, f"L{layer}-E{eid}.bin")


def _cache_store_raw(layer, eid, span):
    path = _cache_path_bin(layer, eid)
    tmp = path + f".tmp{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "wb") as f:
        f.write(span)
    os.replace(tmp, path)


def _cache_load(layer, eid):
    path = _cache_path_bin(layer, eid)
    if os.path.exists(path) and os.path.getsize(path) == EXPERT_SPAN:
        buf = np.memmap(path, dtype=np.uint8, mode="r")
        _bump(expert_disk=1)
        return _slice_expert(buf)
    path = _cache_path(layer, eid)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    _bump(expert_disk=1)
    return {w: (z[w + "_p"], z[w + "_s"]) for w in ("w1", "w2", "w3")}


def fetch_expert_raw(layer, eid):
    """Drop-in for k3loader.fetch_expert_raw: dict w -> (packed u8, scale u8).
    One coalesced Range request over a persistent connection on cache miss."""
    hit = _cache_load(layer, eid)
    if hit is not None:
        return hit
    shard, start, _ = expert_span(layer, eid)
    buf = _range_get(shard, start, EXPERT_SPAN)
    ws = _slice_expert(buf)
    _cache_store_raw(layer, eid, buf)
    _bump(expert_http=1)
    return ws


def _fetch_group(shard, group):
    """group: list of (layer, eid, start) sorted, file-contiguous. One request."""
    base = group[0][2]
    size = group[-1][2] + EXPERT_SPAN - base
    buf = _range_get(shard, base, size)
    if len(group) > 1:
        _bump(coalesced_spans=1)
    out = {}
    for layer, eid, start in group:
        ws = _slice_expert(buf, base=start - base)
        _cache_store_raw(layer, eid, buf[start - base:start - base + EXPERT_SPAN])
        _bump(expert_http=1)
        out[eid] = ws
    return out


def fetch_experts(layer, eids, workers=None, dequant=True, coalesce=True):
    """Parallel fetch of a routed set; signature-compatible with k3loader.
    Coalesces file-adjacent misses into single multi-expert Range requests."""
    workers = workers or MAX_CONNS
    raw, misses = {}, []
    for e in eids:
        hit = _cache_load(layer, e)
        if hit is not None:
            raw[e] = hit
        else:
            misses.append(e)
    if misses:
        spans = sorted((expert_span(layer, e)[1], e) for e in misses)
        shard = expert_span(layer, misses[0])[0]
        groups, cur = [], [(layer, spans[0][1], spans[0][0])]
        for s, e in spans[1:]:
            if coalesce and s == cur[-1][2] + EXPERT_SPAN and len(cur) < MAX_COALESCE:
                cur.append((layer, e, s))
            else:
                groups.append(cur)
                cur = [(layer, e, s)]
        groups.append(cur)
        with concurrent.futures.ThreadPoolExecutor(min(workers, len(groups))) as ex:
            for fut in [ex.submit(_fetch_group, shard, g) for g in groups]:
                raw.update(fut.result())
    if not dequant:
        return {e: {w: (np.ascontiguousarray(p), np.ascontiguousarray(s))
                    for w, (p, s) in ws.items()} for e, ws in raw.items()}
    import torch
    from mxfp4 import dequant_mxfp4
    return {e: {w: torch.from_numpy(dequant_mxfp4(p, s))
                for w, (p, s) in ws.items()} for e, ws in raw.items()}


def close():
    """Close pooled connections (safe to call between batches)."""
    global _httpx_client
    with _pool._lock:
        for _, c in _pool._idle:
            try:
                c.close()
            except Exception:
                pass
        _pool._idle.clear()
    if _httpx_client is not None:
        _httpx_client.close()
        _httpx_client = None
