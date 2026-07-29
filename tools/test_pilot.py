#!/usr/bin/env python3
"""Correctness tests for tools/pilot.py (K3_PILOT router-lookahead prefetch).

    python tools/test_pilot.py            # router mirror + npz hook (fast)
    python tools/test_pilot.py --slots    # also the pread slot-recycling test
                                          # (reads ~4.5 GB of real expert .bin)

None of this times anything; it checks that a prefetch can never change what the
model computes.  The end-to-end gate is still tools/ab_gate.py under K3_PILOT=1.
"""
import hashlib
import os
import random
import sys

os.environ.setdefault("K3_PILOT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a)).hexdigest()


def _digest(ws):
    return {w: (_md5(p), _md5(s)) for w, (p, s) in ws.items()}


# ---------------------------------------------------------------------------
def test_native_int8_capability_fallback():
    """A future wheel/chip without the private MPS op keeps PILOT functional."""
    print("native-int8 capability fallback")
    import pilot

    config = type("Config", (), {
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_experts_per_token": 2,
        "rms_norm_eps": 1e-5,
    })()

    def load(name):
        if name.endswith("gate.weight"):
            return torch.arange(12, dtype=torch.float32).reshape(4, 3)
        if name.endswith("e_score_correction_bias"):
            return torch.zeros(4)
        if name.endswith("post_attention_layernorm.weight"):
            return torch.ones(3)
        raise KeyError(name)

    def packed_must_not_run(_name):
        raise AssertionError("packed loader called with native_int8=False")

    pilot.init(
        config, torch.device("cpu"), load, "model.",
        load_packed=packed_must_not_run, native_int8=False)
    check("PILOT remains enabled", pilot.enabled())
    check("fallback gate is fp32", pilot._W[1].dtype == torch.float32)
    check("fallback is reported", pilot.STATS["gate_dtype"] == "fp32")

    pilot._W.clear()
    pilot._S.clear()
    pilot._B.clear()
    pilot._LN.clear()
    pilot._READY = False
    pilot._BROKEN = False


# ---------------------------------------------------------------------------
def test_router_mirror():
    """pilot._route must reproduce KimiMoEGate.forward, and pilot._rms must
    reproduce KimiRMSNorm.forward, or the recall number means nothing."""
    print("router mirror")
    import kimi_run as kr
    import pilot

    L = 7
    g = f"{kr.PFX}layers.{L}.block_sparse_moe.gate."
    W, B = kr._pilot_load(g + "weight"), kr._pilot_load(g + "e_score_correction_bias")
    pilot._W[L], pilot._B[L] = W, B.to(torch.float32)
    pilot._CFG.update(top_k=kr.config.num_experts_per_token,
                      eps=kr.config.rms_norm_eps, k=kr.config.num_experts_per_token)

    with torch.device("meta"):
        gate = kr.ml.KimiMoEGate(kr.config)
    gate.weight = torch.nn.Parameter(W, requires_grad=False)
    gate.e_score_correction_bias = torch.nn.Parameter(B, requires_grad=False)
    gate.eval()

    torch.manual_seed(0)
    bad = rows = 0
    for _ in range(20):
        h = torch.randn(1, 3, kr.H, device=kr.DEV, dtype=kr.DT) * 0.7
        ref, _w = gate(h)
        mine = pilot._route(L, h.reshape(-1, kr.H).float(), pilot._CFG["k"])[1]
        for a, b in zip(ref.tolist(), mine.tolist()):
            rows += 1
            bad += set(a) != set(b)
    check("_route == KimiMoEGate.forward", bad == 0, f"{bad}/{rows} rows differ")

    ln = kr.ml.KimiRMSNorm(kr.H, eps=1e-5)
    ln.weight = torch.nn.Parameter(torch.rand(kr.H) + 0.5)
    holder = type("Fake", (), {})()
    holder.post_attention_layernorm = ln
    pilot.arm(holder)
    x = torch.randn(1, 2, kr.H)
    y = ln(x)
    check("capture hook fires", pilot._CAP["x"] is x)
    check("capture leaves the norm output untouched",
          torch.equal(y, kr.ml.KimiRMSNorm.forward(ln, x)))
    check("_rms == KimiRMSNorm.forward",
          (pilot._rms(x, ln.weight.float(), 1e-5) - y.float()).abs().max().item() == 0.0)

    # the driver's flatten must stay identical to the old topk_ids.view(-1)
    t = torch.randint(0, 896, (5, 16))
    check("row-major flatten unchanged",
          [e for r in t.tolist() for e in r] == t.view(-1).tolist())

    u = pilot._union([list(range(16)), list(range(10, 26)), list(range(20, 36))],
                     [[1.0] * 16, [2.0] * 16, [3.0] * 16])
    check("_union caps and dedups", len(u) == pilot.MAX_PREFETCH and len(set(u)) == len(u))
    check("_union of one row is that row",
          pilot._union([[3, 1, 2]], [[1.0, 2.0, 3.0]]) == [1, 2, 3])


# ---------------------------------------------------------------------------
def test_npz_hook():
    """A predicted-then-confirmed .npz expert must hand back exactly the bytes a
    cold load would, and a mispredicted one must not leak into anything."""
    print(".npz prefetch hook")
    import fetch_v2
    import pilot

    E = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "k3-experts")
    byl = {}
    for f in os.listdir(E):
        if f.endswith(".npz"):
            L, e = f[1:-4].split("-E")
            byl.setdefault(int(L), []).append(int(e))
    if not byl:
        print("  skip: no .npz experts cached")
        return
    L = max(byl, key=lambda k: len(byl[k]))
    have = sorted(byl[L])[:24]
    if len(have) < 21:
        print("  skip: too few .npz experts on one layer")
        return

    orig = fetch_v2._cache_load
    truth = {e: _digest(orig(L, e)) for e in have}
    pilot.install_npz(fetch_v2)
    check("hook installed", fetch_v2._cache_load is pilot._npz_hook
          and pilot._orig_cache_load is orig)

    pred, actual = have[:12] + have[16:20], have[:16]   # 12 right, 4 wrong
    pilot._PREF[L] = sorted(set(pred))
    pilot.issue_prefetch(L, fetch_v2, pread=True)
    bad = sum(_digest(fetch_v2._cache_load(L, e)) != truth[e] for e in actual)
    check("confirmed + unpredicted loads are byte-identical", bad == 0,
          f"{bad}/{len(actual)} experts differ")
    check("hits counted", pilot.STATS["npz_hits"] == 12, str(pilot.STATS["npz_hits"]))
    n_before = len(pilot._npz_fut)
    pilot.begin_pass()
    check("mispredictions evicted at pass start", n_before == 4 and not pilot._npz_fut)
    check("passthrough for an unpredicted expert",
          _digest(fetch_v2._cache_load(L, have[20])) == truth[have[20]])


# ---------------------------------------------------------------------------
def test_pread_slots():
    """fetch_v2 recycles its pread buffers, so a prefetched slot that is later
    confirmed must still carry its own expert's bytes."""
    print("pread slot recycling")
    import fetch_v2
    if fetch_v2.EXPERT_READ != "pread":
        print("  skip: K3_EXPERT_READ != pread")
        return
    rd, rng = fetch_v2.reader(), random.Random(1234)
    pool = {L: [e for e in range(200) if fetch_v2._has_bin(L, e)] for L in (5, 6, 7, 8)}
    pool = {L: v for L, v in pool.items() if len(v) >= 24}
    if len(pool) < 2:
        print("  skip: not enough local .bin experts")
        return
    layers = sorted(pool)
    bad = checks = 0
    for _ in range(2):
        for L in layers:
            actual = sorted(rng.sample(pool[L], 16))
            wrong = rng.sample([e for e in pool[L] if e not in actual], 5)
            fetch_v2.prefetch_layer(L, sorted(set(actual[:11]) | set(wrong)))
            raw, missing = rd.read_layer(L, actual)
            assert not missing and sorted(raw) == actual
            for e in actual:
                checks += 1
                p = fetch_v2._cache_path_bin(L, e)
                b = open(p, "rb").read()
                want = {w: (_md5(b[o:o + n]), None) for w, o, n, _s in
                        [(nm.split("_")[0], o, n, s) for nm, o, n, s in fetch_v2.LAYOUT
                         if nm.endswith("_p")]}
                got = {w: (_md5(pp), None) for w, (pp, _ss) in raw[e].items()}
                bad += got != want
    fetch_v2.drop_prefetch()
    check("prefetched slots carry the right expert", bad == 0,
          f"{bad}/{checks} experts differ")


if __name__ == "__main__":
    test_native_int8_capability_fallback()
    test_router_mirror()
    test_npz_hook()
    if "--slots" in sys.argv:
        test_pread_slots()
    print(f"\n{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)
