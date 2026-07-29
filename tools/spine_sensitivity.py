#!/usr/bin/env python3
"""Per-ROLE int4-vs-int8 sensitivity study for the resident spine.

Uniform int4 was rejected (10.1x the int8 weight error, see
tools/int4_quality_report.py).  That was a single verdict on 1,253 tensors of 23
different ROLES.  This script asks the finer question: which roles tolerate 4
bits and which do not, so a MIXED spine can spend bits where they matter.

Everything is READ-ONLY on k3-resident/tensors.  Nothing is converted, nothing
is written to any spine directory.

WHAT IS MEASURED, per tensor and per scheme
  rel_fro    ||Wq - W||_F / ||W||_F                       weight space
  rel_l2_iso ||(Wq-W) x||_2 / ||W x||_2, x ~ N(0, I)      output space, isotropic
  rel_l2_act same, but x drawn from a REALISTIC activation model (below)
  post_nl    output error after the nonlinearity the role actually feeds
             (only for roles whose output is immediately squashed)

THE ACTIVATION MODEL.  Every Linear in this net is fed by something whose
per-channel scale is knowable without running the model:

  * fed directly by an RMSNorm  ->  x = norm.weight (*) z,  z ~ N(0,I).
    (q/k/v/g/b/f_a/q_a/kv_a from input_layernorm; router gate, shared-expert
     gate/up and routed_expert_down from post_attention_layernorm; lm_head from
     model.norm; routed_expert_up from routed_expert_norm.)
  * fed by another Linear (+norm)  ->  the input is COMPUTED with the real
    upstream weights read from disk: q_b <- q_a_layernorm(q_a_proj(x)),
    kv_b <- kv_a_layernorm(kv_a_proj(x)[:512]), f_b <- f_a_proj(x),
    shared/dense down_proj <- situ(gate_proj(x), up_proj(x)),
    KDA o_proj <- o_norm.weight-scaled unit vectors gated by sigmoid(g_proj(x)),
    MLA o_proj <- v = kv_b_v(kv_a_layernorm(kv_a(x))) gated by sigmoid(g_proj(x)).

  The one assumption left is that the RMSNorm OUTPUT is per-channel iid
  gaussian, i.e. that the learned norm gain carries the channel geometry.  It is
  a proxy, not a capture: a real forward pass would also carry the residual
  stream's outlier channels.  The int4/int8 RATIO per role is far more robust to
  this than the absolute numbers -- the two schemes see the same reweighting --
  and the ratio is what the policy is built on.  Both isotropic and activation
  numbers are printed so the sensitivity to the model is visible.

  python tools/spine_sensitivity.py                     # full study, ~2 GB read
  python tools/spine_sensitivity.py --max-mb 16         # lighter
  python tools/spine_sensitivity.py --json out.json
"""
import argparse, json, os, sys, time
import collections
import numpy as np

K3 = os.environ.get("DELTAFIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(K3, "k3-resident/tensors")
INV = json.load(open(os.path.join(K3, "k3-meta/tensor_inventory_offsets.json")))
CFG = json.load(open(os.path.join(K3, "k3-meta/config.json")))["text_config"]
P = "language_model.model.layers."
EPS = float(CFG.get("rms_norm_eps", 1e-5))
SITU_BETA = float(CFG.get("activation_situ_beta", 4.0))
SITU_LIN = CFG.get("activation_situ_linear_beta", None)
KV_LORA = int(CFG.get("kv_lora_rank", 512))
HEAD_DIM = int(CFG["linear_attn_config"]["head_dim"])
V_HEAD_DIM = int(CFG.get("v_head_dim", CFG.get("qk_nope_head_dim", 128)))

MLA_LAYERS = {L for L in range(int(CFG["num_hidden_layers"]))
              if P + f"{L}.self_attn.q_a_proj.weight" in INV}


# ----------------------------------------------------------------- I/O ------
def _bf16(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_full(name):
    """Whole tensor as fp32. Only used for the small upstream helpers.
    (o_norm / A_log / dt_bias are stored F32, everything else BF16.)"""
    shape = INV[name]["shape"]
    dt = INV[name]["dtype"]
    if dt == "F32":
        return np.fromfile(os.path.join(RES, name), dtype=np.float32).reshape(shape)
    if dt == "F16":
        return np.fromfile(os.path.join(RES, name),
                           dtype=np.float16).astype(np.float32).reshape(shape)
    raw = np.fromfile(os.path.join(RES, name), dtype=np.uint16)
    return _bf16(raw).reshape(shape)


def read_vec(name):
    return read_full(name).reshape(-1)


def read_rows(name, max_bytes, nchunks=3):
    """A representative row sample: `nchunks` contiguous blocks spread evenly
    over the rows (a single prefix could sit inside one attention head group).
    Returns (W [nr, cols] fp32, rows, cols)."""
    rows, cols = INV[name]["shape"]
    budget = max(1, max_bytes // (cols * 2))
    if budget >= rows:
        return read_full(name), rows, cols
    per = max(1, budget // nchunks)
    starts = [min(rows - per, int(i * (rows - per) / max(nchunks - 1, 1)))
              for i in range(nchunks)]
    outs = []
    for r0 in sorted(set(starts)):
        raw = np.fromfile(os.path.join(RES, name), dtype=np.uint16,
                          count=per * cols, offset=r0 * cols * 2)
        outs.append(_bf16(raw).reshape(per, cols))
    W = np.concatenate(outs, axis=0)
    return W, rows, cols


# --------------------------------------------------------- quantizers -------
def deq_int8(w):
    """Exactly what convert_spine_int8.py writes and kimi_run.py reads back:
    per-row absmax/127 computed in fp32, STORED as fp16, dequantized with the
    fp16 value."""
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return q.astype(np.float32) * s.astype(np.float16).astype(np.float32)


def deq_int4(w, g):
    """Exactly tools/convert_spine_int4.py / int4_loader.py."""
    rows, cols = w.shape
    ng = (cols + g - 1) // g
    cw = ng * g
    if cw != cols:
        w = np.concatenate([w, np.zeros((rows, cw - cols), np.float32)], axis=1)
    wg = w.reshape(rows, ng, g)
    s16 = (np.abs(wg).max(axis=2) / 7.0).astype(np.float16)
    s16[s16 == 0] = np.float16(6e-8)
    s = s16.astype(np.float32)[:, :, None]
    q = np.clip(np.rint(wg / s), -7, 7).astype(np.int8)
    return (q.astype(np.float32) * s).reshape(rows, cw)[:, :cols]


def deq_intb(w, bits, g):
    """Group-wise symmetric b-bit, same quantizer family as int4 (absmax/L,
    scale rounded to fp16 before use). g=0 means one scale per ROW, which is
    what the shipped int8 format uses."""
    rows, cols = w.shape
    L = float(2 ** (bits - 1) - 1)
    if g <= 0:
        s = (np.abs(w).max(axis=1, keepdims=True) / L).astype(np.float16).astype(np.float32)
        s[s == 0] = 1.0
        return np.clip(np.rint(w / s), -L, L) * s
    ng = (cols + g - 1) // g
    cw = ng * g
    if cw != cols:
        w = np.concatenate([w, np.zeros((rows, cw - cols), np.float32)], axis=1)
    wg = w.reshape(rows, ng, g)
    s16 = (np.abs(wg).max(axis=2) / L).astype(np.float16)
    s16[s16 == 0] = np.float16(6e-8)
    s = s16.astype(np.float32)[:, :, None]
    q = np.clip(np.rint(wg / s), -L, L)
    return (q * s).reshape(rows, cw)[:, :cols]


SCHEMES = ["int8", "int4-g32", "int4-g64"]
# the bit-width frontier: what a byte actually buys between 4 and 8 bits
FRONTIER = ["int4-g32", "int4-g64", "int4-g128", "int5-g64", "int6-g64",
            "int6-g128", "int6-r", "int7-r", "int8", "int8-g64"]


def dequantize(w, scheme):
    if scheme == "int8":
        return deq_int8(w)                          # exactly as shipped
    bits = int(scheme[3])
    tail = scheme.split("-")[1]
    return deq_intb(w, bits, 0 if tail == "r" else int(tail[1:]))


def bytes_per_elt(scheme, cols):
    if scheme == "int8":
        return 1.0 + 2.0 / cols
    bits = int(scheme[3])
    tail = scheme.split("-")[1]
    per_g = 2.0 / cols if tail == "r" else 2.0 / int(tail[1:])
    return bits / 8.0 + per_g


# ------------------------------------------------- activation model ---------
def rmsnorm(x, w):
    """x [n, P] column-vectors, w [n] -> normalized over the n axis."""
    r = np.sqrt((x * x).mean(axis=0, keepdims=True) + EPS)
    return (x / r) * w[:, None]


def situ(gate, up):
    a = SITU_BETA * np.tanh(gate / SITU_BETA) / (1.0 + np.exp(-gate))
    if SITU_LIN:
        up = SITU_LIN * np.tanh(up / float(SITU_LIN))
    return a * up


class LayerActs:
    """Realistic input matrices X [cols, P] for every role in one layer."""

    def __init__(self, L, probes, rng, max_bytes):
        self.L = L
        self.pre = P + f"{L}."
        self.probes = probes
        self.rng = rng
        self.max_bytes = max_bytes
        self.is_mla = L in MLA_LAYERS
        self._cache = {}
        H = int(CFG["hidden_size"])
        z = rng.standard_normal((H, probes), dtype=np.float32)
        self.x_in = read_vec(self.pre + "input_layernorm.weight")[:, None] * z
        z2 = rng.standard_normal((H, probes), dtype=np.float32)
        pa = self.pre + "post_attention_layernorm.weight"
        self.x_post = read_vec(pa)[:, None] * z2 if pa in INV else None

    def _mm(self, name, x):
        """Full weight @ x, streamed in row blocks so nothing large is held."""
        rows, cols = INV[name]["shape"]
        out = np.empty((rows, x.shape[1]), np.float32)
        per = max(1, (64 << 20) // (cols * 2))
        for r0 in range(0, rows, per):
            nr = min(per, rows - r0)
            raw = np.fromfile(os.path.join(RES, name), dtype=np.uint16,
                              count=nr * cols, offset=r0 * cols * 2)
            out[r0:r0 + nr] = _bf16(raw).reshape(nr, cols) @ x
        return out

    def _gate_sigmoid(self):
        g = self._cache.get("gsig")
        if g is None:
            g = 1.0 / (1.0 + np.exp(-self._mm(self.pre + "self_attn.g_proj.weight",
                                              self.x_in)))
            self._cache["gsig"] = g
        return g

    def _kv_latent(self):
        v = self._cache.get("kvlat")
        if v is None:
            c = self._mm(self.pre + "self_attn.kv_a_proj_with_mqa.weight", self.x_in)
            v = rmsnorm(c[:KV_LORA], read_vec(self.pre + "self_attn.kv_a_layernorm.weight"))
            self._cache["kvlat"] = v
        return v

    def for_role(self, role, cols):
        """X [cols, probes] for this role, or None -> fall back to isotropic."""
        f = getattr(self, "_x_" + role.replace(".", "_"), None)
        if f is None:
            return None
        x = f()
        if x is None or x.shape[0] != cols:
            return None
        return x

    # --- fed straight by a norm -------------------------------------------
    def _x_self_attn_q_proj_weight(self):      return self.x_in
    _x_self_attn_k_proj_weight = _x_self_attn_q_proj_weight
    _x_self_attn_v_proj_weight = _x_self_attn_q_proj_weight
    _x_self_attn_g_proj_weight = _x_self_attn_q_proj_weight
    _x_self_attn_b_proj_weight = _x_self_attn_q_proj_weight
    _x_self_attn_f_a_proj_weight = _x_self_attn_q_proj_weight
    _x_self_attn_q_a_proj_weight = _x_self_attn_q_proj_weight
    _x_self_attn_kv_a_proj_with_mqa_weight = _x_self_attn_q_proj_weight

    def _x_block_sparse_moe_gate_weight(self):  return self.x_post
    _x_block_sparse_moe_shared_experts_gate_proj_weight = _x_block_sparse_moe_gate_weight
    _x_block_sparse_moe_shared_experts_up_proj_weight = _x_block_sparse_moe_gate_weight
    _x_block_sparse_moe_routed_expert_down_proj_weight = _x_block_sparse_moe_gate_weight
    _x_mlp_gate_proj_weight = _x_block_sparse_moe_gate_weight
    _x_mlp_up_proj_weight = _x_block_sparse_moe_gate_weight

    def _x_block_sparse_moe_routed_expert_up_proj_weight(self):
        # input = routed_expert_norm(sum of expert outputs) -- normed, so the
        # channel geometry is the norm gain.
        w = read_vec(self.pre + "block_sparse_moe.routed_expert_norm.weight")
        return w[:, None] * self.rng.standard_normal((w.size, self.probes), dtype=np.float32)

    # --- fed by an upstream Linear (computed with the real weights) --------
    def _x_self_attn_q_b_proj_weight(self):
        q = self._mm(self.pre + "self_attn.q_a_proj.weight", self.x_in)
        return rmsnorm(q, read_vec(self.pre + "self_attn.q_a_layernorm.weight"))

    def _x_self_attn_kv_b_proj_weight(self):
        return self._kv_latent()

    def _x_self_attn_f_b_proj_weight(self):
        return self._mm(self.pre + "self_attn.f_a_proj.weight", self.x_in)

    def _x_block_sparse_moe_shared_experts_down_proj_weight(self):
        b = self.pre + "block_sparse_moe.shared_experts."
        return situ(self._mm(b + "gate_proj.weight", self.x_post),
                    self._mm(b + "up_proj.weight", self.x_post))

    def _x_mlp_down_proj_weight(self):
        b = self.pre + "mlp."
        return situ(self._mm(b + "gate_proj.weight", self.x_post),
                    self._mm(b + "up_proj.weight", self.x_post))

    def _x_self_attn_o_proj_weight(self):
        gs = self._gate_sigmoid()                      # [12288, P] in (0,1)
        if self.is_mla:
            # attn_output = (attention-weighted average of v heads) * sigmoid(g).
            # A single-token decode with a soft attention distribution keeps the
            # v channel geometry, so use v itself as the pre-gate signal.
            kvb = self.pre + "self_attn.kv_b_proj.weight"
            rows, cols = INV[kvb]["shape"]
            nh = rows // (V_HEAD_DIM * 2)
            full = read_full(kvb) @ self._kv_latent()   # [24576, P], small (512 cols)
            v = full.reshape(nh, 2, V_HEAD_DIM, -1)[:, 1].reshape(nh * V_HEAD_DIM, -1)
            return v * gs
        # KDA: o = o_norm(recurrence out) -- per-head RMSNorm with a learned gain,
        # then a sigmoid gate. Model the recurrence output as per-head isotropic.
        w = read_vec(self.pre + "self_attn.o_norm.weight")
        nh = gs.shape[0] // w.size
        z = self.rng.standard_normal((gs.shape[0], self.probes), dtype=np.float32)
        z = rmsnorm(z.reshape(nh, w.size, self.probes).transpose(1, 0, 2)
                    .reshape(w.size, -1), w).reshape(w.size, nh, self.probes) \
            .transpose(1, 0, 2).reshape(gs.shape[0], self.probes)
        return z * gs


# roles whose output is immediately squashed -- a relative error on the raw
# output overstates what the model actually sees.
def post_nl(role):
    if role.endswith("self_attn.g_proj.weight"):
        return lambda y: 1.0 / (1.0 + np.exp(-y))      # sigmoid gate
    if role.endswith("block_sparse_moe.gate.weight"):
        return lambda y: 1.0 / (1.0 + np.exp(-y))      # router sigmoid scores
    if role.endswith("self_attn.b_proj.weight"):
        return lambda y: 1.0 / (1.0 + np.exp(-y))      # beta = sigmoid
    return None


def role_of(name):
    import re
    return re.sub(r"layers\.\d+\.", "", name).replace("language_model.model.", "") \
             .replace("language_model.", "")


# ------------------------------------------------------------ study ---------
def study_layer(L, acts, roles, max_bytes, out, quiet=False):
    pre = P + f"{L}."
    for role in roles:
        name = pre + role
        if name not in INV or not os.path.exists(os.path.join(RES, name)):
            continue
        W, rows, cols = read_rows(name, max_bytes)
        X = acts.for_role(role, cols)
        realistic = X is not None
        if X is None:
            X = acts.rng.standard_normal((cols, acts.probes), dtype=np.float32)
        Xi = acts.rng.standard_normal((cols, acts.probes), dtype=np.float32)
        Y, Yi = W @ X, W @ Xi
        nY, nYi, nW = np.linalg.norm(Y), np.linalg.norm(Yi), np.linalg.norm(W)
        nl = post_nl(role)
        Ynl = nl(Y) if nl else None
        rec = {"name": name, "layer": L, "role": role, "shape": [rows, cols],
               "rows_used": int(W.shape[0]), "realistic_x": realistic,
               "bytes_int8": rows * cols}
        for s in SCHEMES:
            D = dequantize(W, s)
            E = D - W
            r = {"rel_fro": float(np.linalg.norm(E) / (nW + 1e-30)),
                 "rel_l2_act": float(np.linalg.norm(D @ X - Y) / (nY + 1e-30)),
                 "rel_l2_iso": float(np.linalg.norm(D @ Xi - Yi) / (nYi + 1e-30)),
                 "bpe": bytes_per_elt(s, cols)}
            if nl:
                Dn = nl(D @ X)
                r["post_nl"] = float(np.linalg.norm(Dn - Ynl) / (np.linalg.norm(Ynl) + 1e-30))
            rec[s] = r
            del D, E
        out.append(rec)
        if not quiet:
            b = rec["int8"]
            line = f"L{L:<3d} {role:52s}"
            for s in SCHEMES[1:]:
                a = rec[s]
                line += (f"  {s}: fro {a['rel_fro']:.3f} ({a['rel_fro']/b['rel_fro']:.1f}x)"
                         f" act {a['rel_l2_act']:.4f} ({a['rel_l2_act']/max(b['rel_l2_act'],1e-30):.1f}x)")
            if nl:
                line += f"  [nl g64 {rec['int4-g64']['post_nl']:.4f} vs i8 {b['post_nl']:.4f}]"
            print(line, flush=True)
        del W, X, Xi, Y, Yi


KDA_ROLES = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.g_proj.weight",
             "self_attn.o_proj.weight", "self_attn.b_proj.weight",
             "self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight"]
MLA_ROLES = ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
             "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight",
             "self_attn.g_proj.weight", "self_attn.o_proj.weight"]
MOE_ROLES = ["block_sparse_moe.gate.weight",
             "block_sparse_moe.shared_experts.gate_proj.weight",
             "block_sparse_moe.shared_experts.up_proj.weight",
             "block_sparse_moe.shared_experts.down_proj.weight",
             "block_sparse_moe.routed_expert_down_proj.weight",
             "block_sparse_moe.routed_expert_up_proj.weight"]
DENSE_ROLES = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-mb", type=int, default=32,
                    help="MB of each studied tensor to read (3 spread blocks)")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="override the sampled layer set")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frontier", action="store_true",
                    help="also sweep bit-width 4..8 on a few tensors: what a "
                         "byte buys between int4 and int8")
    ap.add_argument("--shape", action="store_true",
                    help="also report the weight distribution shape (kurtosis, "
                         "outlier mass) per role -- explains the uniformity")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    mb = args.max_mb << 20
    layers = args.layers or [0, 2, 3, 46, 47, 90, 91]
    out, t0 = [], time.time()

    for L in layers:
        acts = LayerActs(L, args.probes, rng, mb)
        roles = (MLA_ROLES if L in MLA_LAYERS else KDA_ROLES)
        roles = roles + (DENSE_ROLES if L == 0 else MOE_ROLES)
        study_layer(L, acts, roles, mb, out)
        del acts

    # lm_head: fed by model.norm
    name = "language_model.lm_head.weight"
    W, rows, cols = read_rows(name, mb)
    d = read_vec("language_model.model.norm.weight")
    X = d[:, None] * rng.standard_normal((cols, args.probes), dtype=np.float32)
    Xi = rng.standard_normal((cols, args.probes), dtype=np.float32)
    Y, Yi = W @ X, W @ Xi
    rec = {"name": name, "layer": -1, "role": "lm_head.weight", "shape": [rows, cols],
           "rows_used": int(W.shape[0]), "realistic_x": True, "bytes_int8": rows * cols}
    for s in SCHEMES:
        D = dequantize(W, s)
        rec[s] = {"rel_fro": float(np.linalg.norm(D - W) / np.linalg.norm(W)),
                  "rel_l2_act": float(np.linalg.norm(D @ X - Y) / np.linalg.norm(Y)),
                  "rel_l2_iso": float(np.linalg.norm(D @ Xi - Yi) / np.linalg.norm(Yi)),
                  "bpe": bytes_per_elt(s, cols)}
    out.append(rec)
    print(f"lm_head  g64 fro {rec['int4-g64']['rel_fro']:.3f} "
          f"act {rec['int4-g64']['rel_l2_act']:.4f} vs int8 act "
          f"{rec['int8']['rel_l2_act']:.4f}", flush=True)

    # ---------------- per-role summary, weighted by real spine bytes --------
    per_role = collections.defaultdict(list)
    for r in out:
        per_role[r["role"]].append(r)
    # real int8 bytes each role occupies across the whole spine
    role_bytes = collections.Counter()
    EX = ("conv1d", "norm", "A_log", "dt_bias", "res_proj",
          "e_score_correction_bias", ".experts.", "vision", "mm_projector")
    for n, t in INV.items():
        if t["dtype"] != "BF16" or len(t["shape"]) != 2 or any(x in n for x in EX):
            continue
        if not os.path.exists(os.path.join(RES, n)):
            continue
        role_bytes[role_of(n)] += t["shape"][0] * t["shape"][1]

    print("\n=== per-role summary (median over sampled layers) ===")
    hdr = (f"{'role':52s} {'GB_i8':>6s} {'i8 act':>8s} {'g64 act':>8s} {'x':>5s} "
           f"{'g32 act':>8s} {'x':>5s} {'g64 fro':>8s} {'x':>5s}")
    print(hdr)
    summary = {}
    for role, recs in sorted(per_role.items(), key=lambda kv: -role_bytes[kv[0]]):
        med = lambda s, k: float(np.median([r[s][k] for r in recs]))
        row = {"gb_int8": role_bytes[role] / 1e9, "n": len(recs),
               "i8_act": med("int8", "rel_l2_act"),
               "g64_act": med("int4-g64", "rel_l2_act"),
               "g32_act": med("int4-g32", "rel_l2_act"),
               "i8_fro": med("int8", "rel_fro"),
               "g64_fro": med("int4-g64", "rel_fro"),
               "g32_fro": med("int4-g32", "rel_fro"),
               "i8_iso": med("int8", "rel_l2_iso"),
               "g64_iso": med("int4-g64", "rel_l2_iso"),
               "realistic": all(r["realistic_x"] for r in recs)}
        row["g64_x"] = row["g64_act"] / max(row["i8_act"], 1e-30)
        row["g32_x"] = row["g32_act"] / max(row["i8_act"], 1e-30)
        row["g64_fro_x"] = row["g64_fro"] / max(row["i8_fro"], 1e-30)
        row["g64_iso_x"] = row["g64_iso"] / max(row["i8_iso"], 1e-30)
        if "post_nl" in recs[0].get("int8", {}):
            row["i8_nl"] = med("int8", "post_nl")
            row["g64_nl"] = med("int4-g64", "post_nl")
            row["g64_nl_x"] = row["g64_nl"] / max(row["i8_nl"], 1e-30)
        summary[role] = row
        print(f"{role:52s} {row['gb_int8']:6.2f} {row['i8_act']:8.4f} "
              f"{row['g64_act']:8.4f} {row['g64_x']:5.1f} {row['g32_act']:8.4f} "
              f"{row['g32_x']:5.1f} {row['g64_fro']:8.4f} {row['g64_fro_x']:5.1f}"
              + ("" if row["realistic"] else "   (iso x)"))
    for role, row in summary.items():
        if "g64_nl" in row:
            print(f"  post-nonlinearity {role}: int8 {row['i8_nl']:.5f} -> "
                  f"g64 {row['g64_nl']:.5f} ({row['g64_nl_x']:.1f}x)")
    if args.shape:
        print("\n=== weight distribution shape (why every role quantizes alike) ===")
        print(f"{'role':52s} {'kurtosis':>9s} {'|w|>3sd %':>10s} {'g64 fro':>8s}")
        for role, recs in sorted(per_role.items(), key=lambda kv: -role_bytes[kv[0]]):
            r = recs[0]
            W, _, _ = read_rows(r["name"], min(mb, 8 << 20))
            x = W.ravel()
            sd = x.std()
            k = float(((x / sd) ** 4).mean())
            frac = float((np.abs(x) > 3 * sd).mean() * 100)
            print(f"{role:52s} {k:9.2f} {frac:10.3f} "
                  f"{summary[role]['g64_fro']:8.4f}")
            del W, x
        print("  (gaussian: kurtosis 3.00, 0.270% beyond 3 sd)")

    if args.frontier:
        print("\n=== bit-width frontier (same quantizer family, real tensors) ===")
        picks = [P + "46.self_attn.g_proj.weight",
                 P + "46.block_sparse_moe.shared_experts.down_proj.weight",
                 P + "47.self_attn.q_b_proj.weight",
                 "language_model.lm_head.weight"]
        picks = [p for p in picks if p in INV]
        print(f"{'scheme':10s} {'B/elt':>7s} {'spine GB':>9s} "
              + "".join(f"{role_of(p).split('.')[-2][:11]:>12s}" for p in picks)
              + f"{'median':>9s} {'x int8':>7s}")
        base = None
        for s in FRONTIER:
            errs, bpe = [], []
            for p in picks:
                W, rows, cols = read_rows(p, min(mb, 16 << 20))
                D = dequantize(W, s)
                errs.append(float(np.linalg.norm(D - W) / np.linalg.norm(W)))
                bpe.append(bytes_per_elt(s, cols))
                del W, D
            m = float(np.median(errs))
            if s == "int8":
                base = m
            print(f"{s:10s} {np.mean(bpe):7.3f} {113.456*np.mean(bpe)/2:9.1f} "
                  + "".join(f"{e:12.4f}" for e in errs)
                  + f"{m:9.4f} {m/(base or m):7.1f}")

    read_gb = sum(r["rows_used"] * r["shape"][1] * 2 for r in out) / 1e9
    print(f"\n(sampled {len(out)} tensors, {read_gb:.2f} GB of weight samples "
          f"+ upstream helpers, {time.time()-t0:.0f}s)")
    if args.json:
        json.dump({"tensors": out, "summary": summary,
                   "role_bytes": dict(role_bytes)}, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
