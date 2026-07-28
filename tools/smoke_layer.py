import sys, os, time, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT+'/tools')
import kimi_run as kr   # reuses config, layers, loaders (no main())
torch.set_num_threads(8)

embed = kr.LazyEmbed()
h = embed([27])  # arbitrary real token embedding
print('embed row:', h.shape, 'norm %.3f' % h.norm())

cache = kr.ml.KimiDynamicCache(kr.config)
with torch.device('meta'):
    layer0 = kr.ml.KimiDecoderLayer(kr.config, 0).eval()
t0=time.time(); kr.materialize_resident(layer0, kr.PFX+'layers.0.'); print('L0 materialize %.1fs' % (time.time()-t0))
br = h.new_zeros(1, 0, kr.H)
t0=time.time()
h1, br = layer0(h, attention_mask=None, position_ids=None, past_key_values=cache, use_cache=True, block_residual=br)
print('L0 (dense KDA) fwd %.1fs  out norm %.3f  finite %s  block_res %s' % (time.time()-t0, h1.norm(), torch.isfinite(h1).all().item(), tuple(br.shape)))
kr.dematerialize(layer0)

with torch.device('meta'):
    layer1 = kr.ml.KimiDecoderLayer(kr.config, 1).eval()
t0=time.time(); kr.materialize_resident(layer1, kr.PFX+'layers.1.'); print('L1 materialize %.1fs' % (time.time()-t0))
kr._step_ctx['layer']=1; kr._step_ctx['step']=-99
t0=time.time()
h2, br = layer1(h1, attention_mask=None, position_ids=None, past_key_values=cache, use_cache=True, block_residual=br)
print('L1 (MoE KDA) fwd %.1fs  out norm %.3f  finite %s' % (time.time()-t0, h2.norm(), torch.isfinite(h2).all().item()))
import k3loader; print(k3loader.cache_report())
