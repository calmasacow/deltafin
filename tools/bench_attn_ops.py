"""Per-op breakdown of ONE KDA layer and ONE MLA layer at T=1 (real shapes,
random weights, each op serialized with torch.mps.synchronize). Also A/Bs the
short-conv formulations and the three recurrence placements.
"""
import os, sys, json, time, statistics
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/tools")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False)
from k3pkg import modeling_kimi_linear as ml
import importlib
CFG = json.load(open(ROOT + "/k3-meta/config.json"))["text_config"]
Cfg = getattr(ml, "KimiLinearConfig", None) or importlib.import_module(
    "k3pkg.configuration_kimi_k3").KimiLinearConfig
config = Cfg(**CFG); config._attn_implementation = "eager"
DEV = torch.device("mps")
H = config.hidden_size
R = int(os.environ.get("R", "15"))


def sync():
    torch.mps.synchronize()


def t(fn, r=R):
    fn(); sync()
    ts = []
    for _ in range(r):
        sync(); t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def set_param(root, dotted, tensor):
    obj = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))


def fill(mod):
    for n, p in list(mod.named_parameters()):
        set_param(mod, n, torch.randn(p.shape, device=DEV, dtype=torch.float32) * 0.02)
    return mod


rows = []
def rec(tag, ms):
    rows.append((tag, ms)); print(f"  {tag:28s} {ms:8.3f} ms", flush=True)


# ============================ KDA ============================
print("=== KDA layer (T=1) ===", flush=True)
with torch.device("meta"):
    kda = ml.KimiDeltaAttention(config, layer_idx=1).eval()
fill(kda)
kda.A_log.data = torch.randn(128, device=DEV) * 0.1
x = torch.randn(1, 1, H, device=DEV) * 0.02
D = kda.num_heads * kda.head_dim
W = kda.conv_size

cache = ml.KimiDynamicCache(config)
cache.conv_states[1] = tuple(torch.randn(1, D, W, device=DEV) * 0.02 for _ in range(3))
cache.recurrent_states[1] = torch.randn(1, 96, 128, 128, device=DEV) * 0.01

ln = fill(ml.KimiRMSNorm(H, eps=config.rms_norm_eps).to(DEV))
rec("input_layernorm", t(lambda: ln(x)))

# attn-res (block_residual with 8 blocks, worst-ish case: 93/12 -> up to 8)
br = torch.randn(1, 8, H, device=DEV) * 0.02
proj = nn.Linear(H, 1, bias=False).to(DEV); proj.weight.data = torch.randn(1, H, device=DEV)
nrm = fill(ml.KimiRMSNorm(H, eps=config.rms_norm_eps).to(DEV))
rec("attn_res(_apply_attn_res)", t(lambda: ml._apply_attn_res(x.view(-1, H), br, proj, nrm)))

rec("q_proj", t(lambda: kda.q_proj(x)))
rec("q+k+v_proj", t(lambda: (kda.q_proj(x), kda.k_proj(x), kda.v_proj(x))))
qp, kp, vp = kda.q_proj(x), kda.k_proj(x), kda.v_proj(x)

cq, ck, cv = cache.conv_states[1]
rec("short_conv x1 (current)", t(lambda: kda.q_conv1d(x=qp, cache=cq, output_final_state=True)))
def three_conv():
    kda.q_conv1d(x=qp, cache=cq, output_final_state=True)
    kda.k_conv1d(x=kp, cache=ck, output_final_state=True)
    kda.v_conv1d(x=vp, cache=cv, output_final_state=True)
rec("short_conv x3 (current)", t(three_conv))

# --- candidate fast short conv (T==1): gather+mul+sum
wq = kda.q_conv1d.weight.view(D, W)
def fast_conv(xp, c, w):
    z = torch.cat([c[:, :, 1:], xp.view(1, D, 1)], -1)
    return F.silu((z * w).sum(-1)).view(1, 1, D), z
rec("short_conv x1 (mulsum)", t(lambda: fast_conv(qp, cq, wq)))
def three_fast():
    fast_conv(qp, cq, kda.q_conv1d.weight.view(D, W))
    fast_conv(kp, ck, kda.k_conv1d.weight.view(D, W))
    fast_conv(vp, cv, kda.v_conv1d.weight.view(D, W))
rec("short_conv x3 (mulsum)", t(three_fast))

# --- candidate: fuse all three into one op
w3 = torch.cat([kda.q_conv1d.weight.view(D, W), kda.k_conv1d.weight.view(D, W),
                kda.v_conv1d.weight.view(D, W)], 0)
def fused3():
    c = torch.cat([cq, ck, cv], 1)             # [1,3D,4]
    xx = torch.cat([qp, kp, vp], -1).view(1, 3 * D, 1)
    z = torch.cat([c[:, :, 1:], xx], -1)
    y = F.silu((z * w3).sum(-1))
    return y.split(D, -1), z
rec("short_conv x3 (1 fused)", t(fused3))

# numerical check
y_ref, c_ref = kda.q_conv1d(x=qp, cache=cq, output_final_state=True)
y_new, c_new = fast_conv(qp, cq, wq)
print(f"   [check] mulsum vs conv1d: y maxdiff={float((y_ref-y_new).abs().max()):.3e} "
      f"cache maxdiff={float((c_ref-c_new).abs().max()):.3e} "
      f"exact={bool(torch.equal(y_ref, y_new))}", flush=True)

q = kda.q_conv1d(x=qp, cache=cq, output_final_state=True)[0].view(1, 1, 96, 128)
k = kda.k_conv1d(x=kp, cache=ck, output_final_state=True)[0].view(1, 1, 96, 128)
v = kda.v_conv1d(x=vp, cache=cv, output_final_state=True)[0].view(1, 1, 96, 128)
g = kda.f_b_proj(kda.f_a_proj(x)).view(1, 1, 96, 128)
beta = kda.b_proj(x).float()
rec("f_a+f_b+b_proj (gates)", t(lambda: (kda.f_b_proj(kda.f_a_proj(x)), kda.b_proj(x))))

from fla.ops.kda import fused_recurrent_kda, _kda_core
S = cache.recurrent_states[1]
kw = dict(A_log=kda.A_log, dt_bias=kda.dt_bias, initial_state=S,
          output_final_state=True, use_qk_l2norm_in_kernel=True,
          use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
          lower_bound=kda.gate_lower_bound)
rec("recurrence (shim: CPU hop)", t(lambda: fused_recurrent_kda(q, k, v, g, beta, **kw)))
kw_cpu = dict(kw, initial_state=S.cpu())    # realistic: state stays CPU-resident
rec("recurrence (hop, cpu state)", t(lambda: fused_recurrent_kda(q, k, v, g, beta, **kw_cpu)))
rec("recurrence (all-MPS)", t(lambda: _kda_core(q, k, v, g, beta, kda.A_log, kda.dt_bias,
    None, S, True, True, True, kda.gate_lower_bound)))
# CPU-resident state variant: inputs already on CPU
qc, kc, vc, gc, bc = (z.cpu() for z in (q, k, v, g, beta))
Sc = S.cpu(); Alc = kda.A_log.cpu(); dtc = kda.dt_bias.cpu()
rec("recurrence (pure CPU, no hop)", t(lambda: _kda_core(qc, kc, vc, gc, bc, Alc, dtc,
    None, Sc, True, True, True, kda.gate_lower_bound)))

o = torch.randn(1, 1, 96, 128, device=DEV)
gg = kda.g_proj(x).view(1, 1, 96, 128)
rec("g_proj (full rank)", t(lambda: kda.g_proj(x)))
rec("o_norm (FusedRMSNormGated)", t(lambda: kda.o_norm(o, gg)))
rec("o_proj", t(lambda: kda.o_proj(o.reshape(1, 1, D))))

def kda_fwd():
    return kda(hidden_states=x, cache_params=cache)
rec("KDA FULL forward", t(kda_fwd))

del kda
torch.mps.empty_cache()

# ============================ MoE gate + shared expert ============================
print("=== MoE gate / shared experts (T=1) ===", flush=True)
with torch.device("meta"):
    gate = ml.KimiMoEGate(config).eval()
    sh = ml.KimiMLP(config=config, intermediate_size=config.moe_intermediate_size *
                    config.num_shared_experts).eval()
    dn = nn.Linear(H, config.routed_expert_hidden_size, bias=False)
    up = nn.Linear(config.routed_expert_hidden_size, H, bias=False)
fill(gate); fill(sh); fill(dn); fill(up)
rec("moe gate (896x7168+topk)", t(lambda: gate(x)))
rec("shared_experts (MLP 6144)", t(lambda: sh(x)))
rec("latent down+up proj", t(lambda: up(dn(x))))
del gate, sh, dn, up
torch.mps.empty_cache()

# ============================ MLA ============================
print("=== MLA layer (T=1, ctx=5) ===", flush=True)
with torch.device("meta"):
    mla = ml.KimiMLAAttention(config, layer_idx=4).eval()
fill(mla)
cache2 = ml.KimiDynamicCache(config)
CTX = 5
cache2.key_cache[4] = torch.randn(1, 96, CTX, 192, device=DEV) * 0.02
cache2.value_cache[4] = torch.randn(1, 96, CTX, 128, device=DEV) * 0.02
rec("q_a_proj+norm+q_b_proj", t(lambda: mla.q_b_proj(mla.q_a_layernorm(mla.q_a_proj(x)))))
rec("kv_a_proj+norm+kv_b_proj", t(lambda: mla.kv_b_proj(mla.kv_a_layernorm(
    mla.kv_a_proj_with_mqa(x)[..., :512]))))
rec("mla g_proj", t(lambda: mla.g_proj(x).sigmoid()))
rec("mla o_proj", t(lambda: mla.o_proj(torch.randn(1, 1, 96 * 128, device=DEV))))
qs = torch.randn(1, 96, 1, 192, device=DEV); ks = cache2.key_cache[4]; vs = cache2.value_cache[4]
rec("eager attn (ctx=5)", t(lambda: ml.eager_attention_forward(mla, qs, ks, vs, None, mla.scaling)))
def mla_fwd():
    c = ml.KimiDynamicCache(config)
    c.key_cache[4] = ks; c.value_cache[4] = vs
    return mla(hidden_states=x, past_key_values=c)
rec("MLA FULL forward", t(mla_fwd))

print("\n=== SUMMARY (ms, median, serialized w/ sync) ===")
for tag, ms in rows:
    print(f"{tag:30s} {ms:8.3f}")
