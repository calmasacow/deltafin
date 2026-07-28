# Deltafin — Running Kimi K3 (2.8T) on Apple Silicon: Implementation Plan

**Machine:** MacBook Pro M1 Max · 32-core GPU · 8P+2E CPU · 64 GB unified RAM · 8 TB internal NVMe (1.2 TB free) · macOS 26.5.2
**Target:** Kimi K3 — 2.8T-param MoE. **WEIGHTS ARE LIVE** (public, ungated, MIT-style license) as of ~15:20 UTC 2026-07-27. **See the Day-0 Addendum at the bottom — config.json + full tensor inventory are in `k3-meta/`, and they revise the RAM ledger.**
**Sources:** deep source study of [colibri](research/colibri) (Apache-2.0) and [ds4/DwarfStar](research/ds4) (MIT), both cloned and **built successfully on this machine**; measured hardware benchmarks; K3 preview coverage incl. the vLLM preview blog.

---

## 0. Executive summary — what is honestly achievable

| Scenario | Decode tok/s | Notes |
|---|---|---|
| Physics ceiling (model magically all in RAM at 2 bpw, GPU-fed at 290 GB/s) | ~18 | unreachable — pool is ~709 GB |
| Speed build (2-bit experts + cache-aware routing + top-12-of-16 + learned pin) | **1.5–3 (central ~2)** | quality must be gated, see §7 |
| Honest build (2–2.5 bpw, true routing, learned pin + lookahead) | 0.7–1.3 | hinges on router skew (§6) |
| Max-quality-that-fits (3.06 bpw) | 0.45–0.7 | fills disk to 1.06/1.2 TB |
| Cold / no cache | 0.25–0.5 | |
| Unmodified ds4/colibri, unported | 0.03–0.15 | why this plan exists |
| Prefill (chunked, streaming) | ~16–40 tok/s | paid once; sessions persist (§5.6) |

Governing equation (validated against every published number from both engines):
`decode time ≈ missed expert bytes ÷ effective SSD bandwidth`, with all compute hidden underneath.
Per-token expert traffic for K3 is ~12.9 GB at ~2 bpw; measured SSD bandwidth is 6.6 GB/s — **hit rate and bytes-per-miss are the only two numbers that matter.** Corroboration: colibri's 2.24 tok/s on M5 Max at 74% hit back-computes to 6.5 GB/s effective SSD — a well-engineered stack attains ~100% of raw disk bandwidth.

Two hard walls found early:
1. **Disk:** K3 ships MXFP4 (4.25 bits/weight) ≈ **1.46–1.49 TB — does not fit in 1.2 TB free.** Sub-4-bit requantization (or expert pruning / tiered precision) is mandatory, not optional.
2. **RAM:** 64 GB leaves a ~40 GB expert cache = **~6% of the routed pool** at 2 bpw. Every tok/s number above ~0.5 is a bet on router skew; the day-one router-trace study (§8) is the go/no-go measurement.

---

## 1. Kimi K3 architecture (preview — re-derive ALL of this from config.json the minute the 401 clears)

- 2.8T total params; MoE with **896 routed experts, 16 active + shared expert(s)**; ~1.8% activation ≈ 50–60B active.
- **93 layers, 3:1 hybrid**: ~70 **Kimi Delta Attention** (KDA — gated-DeltaNet variant, fixed 128×128 state/head, short conv k=4) + ~23 **MLA** full-attention layers (likely NoPE, kv_lora 512 if K2 dims carry).
- **Attention Residuals** (AttnRes): learned softmax attention over previous layers' outputs (Block variant, ~8 blocks). New op — zero local-engine implementations.
- **SiTU** activation (Sigmoid Tanh Unit) — if it replaces SwiGLU inside experts, every fused gate/up/act kernel in both donor engines is wrong as designed. Must be pinned down from `modeling_*.py` before kernel work.
- Gated MLA; sigmoid router (K2 lineage: `noaux_tc`, routed_scaling_factor); **no MTP expected** (every K2-family config: `num_nextn_predict_layers: 0`).
- 1M context; native vision tower (~400M, MoonViT-style) — **out of scope v1** (text-only).
- Weights: **MXFP4** (e2m1 in 32-blocks + shared e8m0 scale = 4.25 bpw), QAT from SFT → MXFP4 *is* full fidelity.
- Tokenizer: near-certain tiktoken, vocab 163840 (whole K2 family + Kimi-Linear share it).
- License: unseen; K2.5 precedent is Modified-MIT (pure MIT below 100M MAU / $20M-mo revenue).
- **The KDA gift:** KV cache ≈ 26 KB/token on only ~23 MLA layers (~3.3 GB @128K ctx) + ~150 MB constant KDA state total. Long context is nearly free in RAM — all spare RAM goes to the expert cache, and session state files are tiny.

Uncertainty that matters: expert byte-size is 18–23 MB at 4.25 bpw depending on whether moe_intermediate ≈ 1500 or 2048 — a ±30% swing on disk fit and cold tok/s. Only config.json resolves it.

## 2. Measured hardware envelope (this machine, this week)

| Resource | Measured | Consequence |
|---|---|---|
| SSD, threaded F_NOCACHE preads (4–8 threads, 1–8 MB) | **6.6 GB/s**; random = sequential at QD≥4 | expert layout on disk is unconstrained; pread pool is mandatory |
| SSD, cold mmap faulting | 0.66–2.0 GB/s | never demand-fault weights (ds4's default path on macOS!) |
| SSD, F_RDADVISE readahead + mmap | 4.96 GB/s | good for prefetch hints, still loses to pread pool |
| QD1 latency | 0.38 ms/1MB … 1.58 ms/8MB | a layer's 16 expert fetches fit inside the previous layer's compute window at QD4–8 |
| GPU read bandwidth (shared MTLBuffer) | 283–290 GB/s | all resident-weight matmuls belong on GPU |
| CPU read bandwidth (8–10 threads) | 105–110 GB/s | CPU NEON amply feeds 6.6 GB/s streamed experts |
| GPU compute | 9.07 TFLOPS fp32 / 8.13 fp16 / 7.42 simdgroup MMA | fp16 is NOT double-rate on M1; prefill is GPU work |
| AMX (Accelerate) | sgemm 1616 GFLOPS; sgemv only 29.9 GB/s | AMX for prefill GEMM tiles (M≥32–64) only; **never** GEMV |
| CPU ISA | DotProd+FHM+FP16, **no i8mm, no bf16** | quant dots = sdot/fmlal hand-NEON; colibri's smmla tiles won't engage |
| Metal limits | maxBufferLength 38.9 GB; workingSet 51.8 GB; wire limit 52.5 GB; Apple7 only (no Metal-4 tensor ops — ds4's MPP path correctly falls back) | split resident pool across buffers; wire ~45–50 GB max |
| Hazard | 28.5 GB swap was in use; quantized weights are incompressible | reboot/clear swap before benchmarks; mlock/wire everything resident; F_NOCACHE everywhere |

## 3. What each engine contributes (technique inventory)

### From ds4 (fork base)
The best Metal streaming architecture in existence for this problem:
- **Resident/streamed split**: all non-routed weights mmap'd + wrapped in a few overlapping `newBufferWithBytesNoCopy` views, wired via **MTLResidencySet**; only routed experts stream.
- **One command buffer per decode token**: whole graph encoded, one wait, one n_vocab logits readback; GPU-resident routing; ~1 sync/token.
- **Expert cache**: direct-indexed [layer][expert] table of gate/up/down MTLBuffers on a single-size-class slab allocator; **selection-based hotness LFU** (counts even misses; halves every 16 tokens) + LRU tiebreak — beat hit-count LFU in their testing.
- **Streamed-expert zero-copy dispatch**: pread pool (default 9 threads) reads from the GGUF fd *directly into MTLBuffer contents*; kernels consume per-expert `gpuAddress` tables; **masked dispatch** computes resident experts while misses stream.
- **Fused kernels**: pair_swiglu (gate+up+act+route-weight, one dispatch) and sumK (down-proj + weighted accumulation) → ~2 dispatches per MoE layer per token; llama.cpp-lineage flash attention (fp16 storage, fp32 accumulation, split-K decode).
- **Sessions**: KVC disk cache keyed by SHA1 of rendered bytes; full-logits-in-snapshot; byte-exact tool-call replay (rax); zero-prefill resume. Port verbatim.
- **DVFS keep-alive kernel** (from its TP code): M-GPUs down-clock in stall gaps (~1.7× kernel-time inflation measured) — run the spinner during SSD-miss stalls.
- **Quant recipe**: routed-only IQ2_XXS gate/up + Q2_K down + imatrix; everything else Q8_0; quality gated by **avg_nll vs official API continuations** (not perplexity).
- Bench methodology (ds4-bench): frontier suffix-prefill timing, first-token vs steady-state split, CSV.

### From colibri (levers to graft on)
- **PILOT router lookahead**: run layer L+1's router on L's post-attention state → **71.6% top-K recall one layer ahead** (vs 41.3% previous-token); dedicated prefetch thread + lock-free ring; LFRU guard so speculation can't evict warm demand experts.
- **Learning cache**: persistent per-expert usage counters (`.coli_usage`) → AUTOPIN hottest experts at startup, mlock'd; "gets faster the more you use it" (1.11→1.83 tok/s across runs).
- **CACHE_ROUTE** (arXiv:2412.00099): keep true top-J(2), fill remaining slots preferring resident experts within top-M(12). Measured **+39%** (2.4→3.33 tok/s) on GB10. Routing-side, lossy, telemetry built in (overlap%, KL).
- **DIRECT=1 discipline**: F_NOCACHE twin fds; buffered reads measured ~2× slower end-to-end (breaks zero-copy GPU slab feeding, pollutes page cache).
- **The −39% OMP trap**: CPU active-spin steals the shared SoC power budget and throttles the GPU (attention kernel 76→223 s for identical work). Passive waits + a small async I/O pool are *complementary and both required*.
- **Dual-SSD mirror**: deterministic hash routing over byte-identical (even partial) mirrors, bandwidth-weighted; +aggregate read bandwidth.
- **Memory budget math**: cap-for-RAM with an explicit **2.5 GB page-cache reserve** (starving it dropped buffered reads 800→180 MB/s), RSS guard shrinking cache live.
- **Conversion pipeline skeleton**: `c/tools/convert_fp8_to_int4.py` — shard-by-shard download→convert→delete (peak disk = 1 shard + output), resume manifests, multi-stream Range downloader, `--min-free-gb` guard, **e2m1 FP4 LUT already present** (verified vs ml_dtypes).
- Spec-decode machinery (MTP chain, Leviathan rejection) — likely moot for K3 (§6), but the grammar-forced spans idea stays interesting for JSON tool output.

## 4. Where we beat both engines on this machine (measured deltas)

1. **Fix the macOS streaming path (3–10×)**: ds4's only decode readahead is `posix_madvise(WILLNEED)` — **a no-op on XNU**; its cold streaming runs at 0.66–2 GB/s. colibri single-threads demand reads (~5 GB/s). We use a 4–8-thread **F_NOCACHE pread pool** at expert granularity = **6.6 GB/s**, saturating the device. This is the single biggest win and it's pure I/O plumbing.
2. **One pread per expert**: ds4 issues 3 preads/expert (gate/up/down at distant offsets). Our converter lays each expert's three matrices file-adjacent, 4 KiB-aligned (colibri's trick) → one ~5–8 MB read per miss.
3. **Never break the command buffer for streaming**: ds4's streaming decode falls back to per-layer CB waits. We use its own TP-gate pattern instead — `encodeWaitForEvent` after each router, I/O thread signals when preads land; GPU never drains.
4. **Decode by bandwidth domain**: GPU (290 GB/s) runs attention + dense + shared + cached-hot experts; CPU NEON (105 GB/s ≫ 6.6 GB/s arrival) runs cold streamed experts → streaming compute fully hidden. Neither engine splits this way.
5. **E-core discipline**: pin I/O + staging threads to the 2 E-cores (QoS utility), keep 8 P-cores + GPU on compute; never spin (the −39% trap is proportionally worse on a 10-core part).
6. **Prefill MMA**: colibri's prefill is GEMV-only (no simdgroup_matrix anywhere); we use ds4-style 64×32-tile mul_mm_id with in-kernel dequant on GPU + AMX (1.6 TFLOPS sgemm) for CPU-side tiles at M≥32.
7. **Kernel micro-tuning**: explicit `#pragma clang loop unroll_count(16)` took a measured kernel from 4.0→9.1 TFLOPS — Metal under-unrolls FMA chains on M1.
8. **Top-k reduction as a first-class knob**: drop lowest-score-mass experts (top-12-of-16 → ×1.33, top-8 → ×2.0 on disk-bound decode). Graduated, measurable, composable — stronger than any cache trick, and neither engine exposes it as a tuned default.
9. **Tiered/dual-precision storage**: hot experts (router-trace ranked) kept at native MXFP4 (QAT-exact); cold tail at 2–2.5 bpw; total sized to ~1.05 TB. Quality loss concentrates on rarely-used experts. Optional cfse/rANS on the cold tier only if decode >1 GB/s/E-core benchmarks out (expect ≤1.1× on 2-bit-class data — verify before betting).
10. **KDA-aware memory ledger**: ~70 of 93 layers need no KV at all → reassign ~25 GB (vs an all-MLA design) to the expert cache; persist the ~150 MB state for instant session resume.

## 5. Engine architecture (the build)

**Decision: fork ds4** (`research/ds4`) — its Metal runtime, streaming cache, session layer, and quality methodology are the strongest base — and graft colibri's PILOT / AUTOPIN / CACHE_ROUTE / mirror / budget-math on top. Both licenses (MIT / Apache-2.0) are compatible with attribution.

### 5.1 Raise ds4's hard caps (day-one mechanical work)
`DS4_MAX_LAYER` 79→96 · streaming cache [80][384]→[96][896] (switch victim scan to per-layer clock or heap — full sweep is now 86k entries) · `DS4_METAL_MAX_ROUTED_EXPERT_USED` 8→16 · generate `sumN` Metal kernels from shape (kill sum6/sum8 hardcoding) · top-K select for E=896 (two-level reduction; colibri's `r_top8_par` caps at E≤256) · hotlist entry format u16→u32 · new `kimi3` GGUF/metadata namespace + shape table + `exit(1)` validation in ds4's style.

### 5.2 New ops (correctness-first, in dependency order)
1. **MXFP4 dequant** in-shader + NEON (e2m1 nibble LUT + e8m0/32 scale; ggml has had a native MXFP4 type since gpt-oss — steal layout).
2. **KDA decode step on CPU NEON first** — state is 128×128/head, trivial per-token FLOPs; port from llama.cpp PR #18755 (Kimi-Linear KDA, merged, backend-agnostic) + fla-org Triton as reference. Metal chunked-prefill kernel in week 2.
3. **SiTU** activation (definition from `modeling_*.py`; patches pair_swiglu).
4. **Parameterized MLA fused decode** for the ~23 full-attention layers — ds4's GLM MLA kernel family (kv_lora_rms_norm / qk_lowrank / value_project / compact-KV) is the template; DK=512 flash-attention instantiation already matches.
5. **AttnRes** — trivial FLOPs (per-layer pseudo-query dot over ~8 block vectors), but get it right from the reference repo (MoonshotAI/Attention-Residuals) before anything runs.
6. Router: sigmoid + bias, top-16, Quantile-Balancing semantics as shipped.

### 5.3 Streaming/decode dataflow (per token)
GPU encodes whole token in one CB → per layer: router runs on GPU → shared-event readback of 16 selected ids → hits dispatch immediately (masked kernels, GPU); misses fan out to the F_NOCACHE pread pool (E-cores) landing directly in slab MTLBuffer contents → `encodeWaitForEvent` releases the layer's miss-complement dispatch → PILOT thread meanwhile runs layer L+1's router on L's state and pre-issues its misses. DVFS keep-alive spinner runs during stalls. Selection-hotness LFU eviction; AUTOPIN from persisted usage stats at startup.

### 5.4 Memory ledger (auto-generated from config.json on drop day)
Provisional: dense/attention/KDA/shared/embeddings resident ~10–15 GB (int4/int8 mix) + KV 1.5–3 GB + OS ~6 + page-cache reserve 2.5 + slack 2 → **~38–42 GB wired expert cache** (~2,000–4,900 experts). Raise `iogpu.wired_limit_mb` via sudo; respect the 38.9 GB per-buffer cap; watch `vm.user_wire_limit` (52.5 GB); DISPATCH memory-pressure source shrinks the cache live (colibri's RSS-guard idea).

### 5.5 Conversion pipeline (adapt colibri's converter — do NOT write from scratch)
`convert_fp8_to_int4.py` gains: MXFP4 e8m0-per-32 branch beside its existing NVFP4 e2m1 LUT · K3 tensor classifier (KDA/conv/AttnRes/MLA/router/vision skip-list) · un-stack fused `[E,…]` expert tensors into **file-adjacent, 4 KiB-aligned gate/up/down triplets** (16 KiB slab boundaries for Metal no-copy) · tiered output precision by router-trace rank · raised `--min-free-gb` (end-state margin <150 GB). Shard-by-shard download→convert→delete keeps peak disk = 1 shard + growing output. Staging option: the LAN Mac (2 TB free over SMB) for the raw shards. Run under `caffeinate`.
**Unknown that may dominate week one: raw download bandwidth for 1.5 TB.** Measure today on a K2.5 shard (at 40 MB/s it's 10+ days; at 1 GB/s ~5 h).

### 5.6 Prefill + sessions
Layer-major streaming prefill (ds4 pattern) with capped in-flight bytes — full-expert-set page-in is ~18 GB/layer for K3, so the 2-layer double-buffer must shrink to a bounded window. Ceiling ~16–40 tok/s; a cold agentic prompt costs minutes — which is why we port ds4's KVC session discipline verbatim (SHA1-of-rendered-bytes keys, byte-exact template replay) + persist the tiny KDA state: **every prefix is paid once, ever.**

### 5.7 Correctness harness (before believing any token)
- Per-layer PyTorch oracle streaming one layer's weights at a time (~18–23 GB transient — fits); activation fixtures for KDA/AttnRes/SiTU/MLA/router. Read `modeling_*.py` (trust_remote_code) before executing it.
- Tokenizer: tiktoken loader or tiktoken→tokenizer.json conversion; 10k-sample round-trip vs `transformers`; byte-exact chat-template tests (K3 template hand-ported to C like colibri's GLM one).
- End-to-end: ds4's avg_nll-vs-official-API gate (~$10–50 of K3 API) run on **every quality-costing lever separately** — 2-bit requant, CACHE_ROUTE substitution, top-k cut, pruning — because the levers compound and nobody has measured the stack.
- Keep a serial deterministic validation config (colibri's lesson: parallel reductions flip argmax within ~7 tokens).

## 6. Feasibility math (the honest version)

- Streamed bytes/token: **12.9 GB @2.06 bpw · 15.6 @2.5 · 19.2 @3.06 · 26.6 @4.25**. Resident-weight traffic (~7–13 GB/token from RAM) costs 25–47 ms at GPU bandwidth — never the bottleneck.
- tok/s = 6.6 GB/s ÷ ((1−h) × bytes/token). At 2.06 bpw: h=50% → 1.0 · h=70% → 1.7 · h=90% → 5.1 (unreachable honestly — needs ~25–40% of the pool cached, i.e. a 256 GB machine).
- Calibration points: GLM-5.2 hit 72–75% with 12.7–31% of the pool cached. K3 gives us ~6% cache fraction on a router explicitly trained for load balance → honest band **h ≈ 35–60%**.
- **MTP/draft speculation is ~a wash when disk-bound** (k verified tokens route near-disjoint expert sets → streamed bytes scale with accepted tokens). Skip in v1; revisit only if a high-hit regime materializes. (Also: no K2-family model shipped an MTP head.)
- Two-Mac scaling: ds4's Thunderbolt TP needs both expert halves resident (impossible at 2.8T on 2×64 GB); TCP pipeline mode makes decode *slower*. Not a lever for this model. (The LAN Mac is still useful as download staging.)
- Fantasy numbers to refuse: ≥5 tok/s decode, MXFP4-native local storage, unmodified-engine anything.

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Router flatness (hit ≈ cache fraction) | decode ~0.55 tok/s regardless of engineering | day-one router-trace study; if top-5k experts carry <40% of routing mass → CACHE_ROUTE + pruning become mandatory, or accept batch-style usage |
| Sub-MXFP4 requant of a QAT'd model degrades disproportionately | quality collapse below usefulness | rehearse on K2.5 (same QAT lineage) first; tiered precision; avg_nll gate per lever |
| SiTU/AttnRes/KDA-variant mis-implementation | silent garbage (no crash) | oracle fixtures before kernels; llama.cpp #18755 + official repos as references |
| Expert size 18 vs 23 MB (config unknown) | ±30% on disk fit & speed; 23 MB may push even 3 bpw over disk | no irreversible conversion until config.json read |
| HF repo gated (401 pattern) | scripted download blocked | pre-provision HF_TOKEN + hf_transfer; check ModelScope mirror; human click-through if needed |
| Download bandwidth | week-one schedule | measure today on K2.5; consider LAN-Mac staging |
| Swap/compressor fighting streaming | halves effective SSD bandwidth | clean reboot, wire resident set, F_NOCACHE everywhere, pressure-source cache shrink |
| Thermal/sustained-I/O derating over hours | unknown | long-duty bench during conversion (which is itself a sustained-I/O rehearsal) |

## 8. Runbook

**Now (before the 401 clears) — rehearse everything on K2.5:**
1. `hf auth login`; test `hf_transfer` against a gated repo; probe ModelScope for a K3 mirror.
2. Download one K2.5 shard → measure real bandwidth; benchmark converter MB/s on it → publish hours-to-first-token estimate.
3. Write + test the MXFP4 dequant branch against `ml_dtypes` fixtures; tiktoken round-trip harness on K2.5's tokenizer.
4. Requant K2.5 routed experts to 3.06/2.5/2.06 bpw tiers; avg_nll gate vs Moonshot API → the requant-quality curve for the same QAT lineage.
5. Mechanical ds4 fork work (§5.1 caps) — no K3 numbers needed.
6. Clean the machine: reboot, clear the 28.5 GB swap, set `iogpu.wired_limit_mb`.

**T+0 (release watcher fires — armed, polling every 60 s):**
1. Read LICENSE + gate terms; pull `config.json`, `modeling_*.py`, tokenizer files, chat template (a few MB) — **re-derive every number in §1/§6; regenerate the RAM ledger.** Diff against the vLLM K3 tree.
2. Pin down SiTU / AttnRes / KDA-variant / MTP-presence from the modeling code.
3. **Router-trace study = the day-one deliverable, not kernels**: per-layer oracle over ~1M calibration tokens → expert-frequency histogram + cross-layer co-activation → hit-rate-vs-pin-GB curve → decides 0.3 vs 2 tok/s and whether pruning is mandatory.
4. Start the shard-by-shard download+convert (tiered precision from the trace ranking), under caffeinate, with resume manifests.

**T+1..3 (while conversion runs):** oracle fixtures → CPU NEON path for KDA/SiTU/AttnRes/MLA → first tokens on CPU decode (slow but correct, gated vs oracle).

**Week 1–2:** Metal kernels in §5.2 order → streaming dataflow §5.3 → PILOT + AUTOPIN + hotlists → top-k knob + CACHE_ROUTE with quality telemetry → ds4-bench harness numbers (frontier prefill + steady-state decode, CSV) → tune toward the 1.5–3 tok/s band.

---

## Day-0 Addendum (written ~15:30 UTC, minutes after the drop)

Fetched without downloading the model: `config.json`, LICENSE, `model.safetensors.index.json`, and **all 96 shard headers via Range requests** → complete 497,220-tensor inventory in [k3-meta/tensor_inventory.json](k3-meta/tensor_inventory.json).

### Confirmed architecture (supersedes §1 estimates)
- 93 layers (layer 0 dense, intermediate 33792); **896 experts, top-16, `num_shared_experts: 2`** (fused as one 6144-intermediate tensor set per layer); sigmoid router + e_score_correction_bias, `noaux_tc`, group count 1 (ds4's validation constraint holds!). **`num_nextn_predict_layers: 0` — no MTP, confirmed.** 1M context, vocab 163840, tiktoken.
- **69 KDA layers + 24 MLA layers** (config `full_attn_layers` is 1-based; classify by tensor names). KDA: 96 heads × 128, q/k/v/g projections [12288,7168] + o [7168,12288], short convs k=4, 128-rank gate LoRA (`f_a/f_b`), `A_log`/`dt_bias` — DeltaNet-family, llama.cpp #18755 remains the reference. MLA: q_lora 1536, kv_lora 512, nope 128 + rope 64, v_head 128, NoPE, **output gate** (`mla_use_output_gate`).
- **Stable LatentMoE confirmed and it's great news**: each MoE layer has shared `routed_expert_up_proj` [7168→3584] / `down_proj` [3584→7168] / norm; **all 896 routed experts operate in the 3584-dim latent** as w1/w3 [3584→3072] + w2 [3072→3584] (Mixtral-style naming, `block_sparse_moe.experts.N.wX`). Per expert: 3 × 5,505,024 B MXFP4-packed U8 + 3 scale tensors (e8m0, 1 B per 32-group) ≈ **17.5 MB/expert**. Experts are **individual tensors, not stacked** — the converter's un-stacking step vanishes; it's requant + relayout + align only.
- **AttnRes is tiny, exactly as hoped**: per layer one `self_attention_res_proj` [1,7168] + one `mlp_res_proj` [1,7168] (+norms; block size 12) — two pseudo-query dots per layer, negligible FLOPs, correctness-only work.
- **SiTU**: `hidden_act: situ`, β=4.0, linear_β=25.0 — pull the exact formula from `modeling_kimi_linear.py` before writing pair-activation kernels.
- Quantization as shipped: routed experts MXFP4 (group 32, e8m0 uint8 scales); **everything else bf16** (attention, shared experts, latent projections, router, dense layer, embeddings/head).

### Revised sizing (the one plan-changing number)
- **Total: 1.561 TB** (routed weights 1361 GB + scales 85 GB + resident 114.4 GB + vision 0.9 GB). Disk wall confirmed, slightly worse than estimated.
- **Resident (non-routed, non-vision) ships as 113.5 GB bf16 ≈ 57B params** — attention 72.4 GB (the 96-head KDA layers are ~885 MB each), shared experts 24.3 GB, latent projections + AttnRes + norms ~9.5 GB, embeddings + head 4.7 GB. §5.4's "10–15 GB resident" was wrong by ~2×: **the resident set must itself be quantized (~4–4.5 bpw → ~32 GB; none of it was QAT'd — quality must be validated on the K2.5 rehearsal)**.
- Revised RAM ledger: resident ~32 GB + MLA KV @128K ~3.5 GB + KDA state ~220 MB + OS 6 + page-cache reserve 2.5 + slack 2 → **expert cache ~17–20 GB ≈ 2,000–2,400 experts ≈ 2.4–2.9% of the pool** (was 5.9%). A ~3.5 bpw resident mix (IQ3-class shared/latent + int4 attention) buys back ~7 GB of cache and is worth testing.
- Revised decode math at 2.06 bpw routed (~8.5 MB/expert streamed): 12.5 GB/token cold; honest hit band drops to ~25–45% → **honest build ~0.7–1.0 tok/s; speed build (top-12 cut + CACHE_ROUTE + learned pin) ~1.3–2.2 tok/s**. The router-trace study (§8) is now even more decisive, and expert pruning moves from "optional" toward "likely".
- Disk after conversion: 2.06 bpw routed (~700 GB) + resident (~32 GB) ≈ **735 GB — fits with ~450 GB slack**, enough for a native-MXFP4 hot shard of the ~10–15k hottest experts (~175–260 GB) for QAT-exact quality on cache hits.
- License: permissive MIT-style ("without restriction" grant). Repo is public and ungated — scripted download works.

*Working assets: both engines build on this machine (`research/ds4/ds4` with Metal; `research/colibri/c/colibri` with METAL=1). Study corpus: 8 agent reports in `research/study/`. K3 metadata + full tensor inventory: `k3-meta/`. The HF release watcher fired and has ended.*

---

## Day-0 Evening Addendum — IT RUNS, and the optimization scoreboard

**K3 generated correct text on this machine** via the lazy-K3 driver ([tools/kimi_run.py](tools/kimi_run.py)): `"The capital of France is"` → `" Paris. The Eiffel Tower is located"` — 8 greedy tokens, all correct English, from the full 93-layer/896-expert model. Stack: Moonshot's own modeling code + pure-PyTorch fla shim ([tools/fla/](tools/fla), chunk-vs-step consistency 1e-9) + 114 GB local resident spine + on-demand HTTP expert streaming with disk cache + fused NEON MXFP4 GEMV ([tools/libmxfp4gemv.dylib](tools/fused_gemv.c), bit-exact). First run: prefill 40 min, ~20 min/token, 66% of time in expert fetch. Expert cache after day 0: ~9.6k experts / 168 GB (11.6% of model).

**Optimization rounds (all measured on real K3 weights, this machine):**
| idea | verdict | measured |
|---|---|---|
| Fused MXFP4 dequant+GEMV (NEON TBL + e8m0 exponent-add) | ✅ shipped | bit-exact; 15.8 GB/s@4T; 41–123× the numpy path |
| HTTP fetch v2 (1 range/expert + keep-alive CDN pool) | ✅ shipped | 6.4× under live contention; expert latency 6.5–8.8 s → 1.1 s; HTTP/2 *loses* 1.7× — use HTTP/1.1 keep-alive |
| torch-MPS resident compute | ✅ shipped | logits identical to 4 decimals; resident compute 903→117 s (7.8×) |
| int8 resident spine (per-row symmetric) | ✅ shipped | 114.4→60 GB/pass; argmax + top-5 order preserved, top-logit shift 0.07% |
| Metal fused MXFP4 GEMV (16-expert batches) | ✅ prototype validated | 150 GB/s sustained = 9.5× CPU kernel; per-expert dispatch unusable (29 GB/s) — batch per layer; zero-copy confirmed |
| Buffered + F_RDADVISE readahead | ⚠ conditional | +18% over F_NOCACHE pool (7.0 GB/s ≈ SSD ceiling); re-test under wired-memory pressure |
| Latent-expert SVD sketches (approx-on-miss / shared basis) | ❌ dead | spectra flat (rank-128 = 13% energy, 93% output error); cross-expert basis at random floor |
| APFS/zstd compression of experts | ❌ dead | payload entropy 7.51 bits/B; decmpfs refuses; kernel decompress 1.7 GB/s < SSD |
| MTP/draft speculation | ❌ dead (confirmed) | config: `num_nextn_predict_layers: 0`; disk-bound wash regardless |

**First real K3 routing data** ([tools/analyze_trace.py](tools/analyze_trace.py), 8-step trace): consecutive-token expert overlap **39.7%** (colibri's GLM figure: 41.3% — previous-token prefetch generalizes); top-16-of-896 carries 40.8% of per-layer routing mass in-sample (~5× uniform). Longer traces needed for the full skew curve; every run appends.

**Next:** longer traces → router surrogate + AUTOPIN-style pinning; Metal expert kernel into the driver; then the ds4-fork C engine per §5 with all validated pieces. Still on the shelf: ANE spine (CoreML stateful), Thunderbolt cold-tier via the LAN Mac, K2.5 requant-quality rehearsal for the sub-4-bit disk decision.
