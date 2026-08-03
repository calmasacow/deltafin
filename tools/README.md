# Tools directory boundary

Deltafin's supported product is the compiled `deltafin` executable built from
the root Cargo workspace. There is no Python installation step, Python
fallback, shell launcher or interpreted build step in a supported command.

## Compiled production inputs

`native/deltafin-native-build/src/lib.rs` owns the shared declarative
production and provider-test source graph. `native/deltafin/build.rs` is only
a one-line Cargo adapter into that Rust crate. In addition to Rust and the
reviewed provider sources under `native/provider_gate`, the graph selects only
these files from this directory:

- `fused_gemv.c`
- `fused_gemv_batch.c`
- `neon_compat_x86.h`
- `metal_moe_abi.h`
- `metal_moe.mm`
- `metal/moe_mxfp4.metal`
- `gpu_runtime_compat.h`
- `cuda_moe_kernels.cu`

They are compiled into the in-process provider. Users do not launch them, and
the build never discovers source files by executing code from this directory.

## Supported replacement map

| Former responsibility | Supported compiled command |
|---|---|
| Inference or chat | `deltafin run` |
| OpenAI-compatible server | `deltafin serve` |
| Complete K3 and DSpark install | `deltafin setup` |
| Pinned K3 metadata audit/install | `deltafin setup-k3` |
| Pinned DSpark audit/install | `deltafin setup-dspark` |
| Optional Qwen install | `deltafin setup-qwen` |
| Resident or expert weight transfer | `deltafin fetch-weights` |
| Resident-spine preparation | `deltafin pack-spine` or `deltafin convert-spine-int8` |
| Route-based expert warming and legacy NPZ migration | `deltafin warm-expert-cache` |
| Lossless expert scale sidecars | `deltafin convert-experts-scale4` |
| Reproducible measurement | `deltafin benchmark` |
| Installation/provider self-test | `deltafin doctor` |
| Safe source upgrade | `deltafin upgrade` |
| Production build | `cargo build --locked --release` |

## Quarantined migration references

The following old roots may remain in a development checkout only because they
contain independent reference semantics, uncaptured historical work, or both.
They are not supported
commands, imports, fallbacks, installers or runtime dependencies. They are not
required to build or run `deltafin`, and no public documentation invokes them.

| Quarantined file | Native owner | Why it has not been deleted yet |
|---|---|---|
| `kimi_run.py` | `deltafin run` and the native target engine | Independent full-target numerical oracle used for exact token parity |
| `serve_openai.py` | `deltafin serve` | Local ignored historical reference; native HTTP, SSE, request, memo and publication semantics are covered by Rust tests |
| `setup_dspark.py` | `deltafin setup-dspark` | Historical DSpark payload-admission semantics |
| `fetch_v2.py` and `fetch_spine.py` | `deltafin fetch-weights` and native lazy expert storage | Historical HTTP-range and cache-layout implementation |
| `convert_experts_scale4.py` | `deltafin convert-experts-scale4` | Historical byte-format producer and Metal comparison oracle |

`kimi_run.py` directly reaches `runtime_platform.py`, `packed_q8.py`,
`quality_policy.py`, `routing_record.py`, `mps_route_mailbox.py`, `k3loader.py`,
`apple_silicon.py`, `kv_cache.py`, `spine_fast.py`, `spine_io.py`,
`attn_fast.py`, `mla_latent.py`, `aten_tape.py`, `pilot.py`, `grouped_moe.py`,
`spec_decode.py`, `shortconv3_mps.py`, `spine_cache.py`, `k3pkg/` and `fla/`.
Its optional historical branches also reach `fast_moe.py`,
`fast_moe_batch.py`, `cuda_moe.py`, `metal_moe.py`, `fetch_v2.py`,
`dspark_model.py`, `dspark_runtime.py`, `dspark_q8.py`, `model_source.py` and
`universal_draft.py`. That entire graph is reference-only; none is reachable
from the Cargo production graph.

The old server-only graph—`server_kv_cache.py`, `server_tokenizer.py`,
`response_memo.py` and its optional drafting helpers—is likewise quarantined.
Its production responsibilities now live in `engine.rs`, `tokenizer.rs`,
`chat.rs`, `openai/server.rs` and `openai/response_memo.rs`.

All `test_*.py`, `bench_*.py`, `probe_*.py`, `validate_*.py` and the remaining
one-off analysis/conversion scripts (including the ignored
`build_spine_layer_pack.py` experiment) are development evidence. They may be
useful when auditing old experiments, but they have no compatibility promise
and must never be cited as installation or usage commands.

## Retired roots

These obsolete roots were retired from the published source after their
behavior was covered by native commands and native tests. Ignored local copies
may remain in an existing development checkout as historical evidence:

- `build_native.py` and its Python-only tests (the shared
  `deltafin-native-build` crate owns target admission, exact source and flag
  selection, CPU ISA dispatch, Metal metallib compilation, CUDA toolkit/ABI and
  architecture checks, native-tool validation, the interpreter-denial guard,
  provider ABI tests and linkage policy; Cargo and the safe upgrader own staged
  publication). Its former loose `libmxfp4*`, `libk3*` and
  `libdeltafin_hybrid` outputs served only the retired Python runtime boundary;
  production links the same reviewed kernels into the single executable.
- `bench.py` (`deltafin benchmark` owns the campaign schema, interleaving,
  exact-output and drafter contracts, bounded native-runner capture, durable
  evidence and summary statistics; historical stdout parsing is deliberately
  not a production compatibility surface); its Python-only test root was
  retired with it
- `convert_spine_int8.py` (`deltafin convert-spine-int8` owns the byte-exact
  format, NumPy-gold rounding contract, authenticated resumability, torn-pair
  recovery and atomic publication)
- `fetch_experts_all.py` (`fetch-weights` owns layer selection, dry runs and
  resumable expert transfer)
- `warm_expert_cache.py` and `convert_npz_cache.py` (`warm-expert-cache` owns
  trace ranking, bounded fetches and authenticated lossless NPZ migration)
- `selftest.py` (`doctor` and the native provider test suite own model/provider
  validation)
- `setup_draft.py` and `test_setup_draft.py` (`deltafin setup-qwen` owns both
  pinned Qwen rosters, data-only configuration admission, authenticated resume,
  legacy payload adoption, transactional publication and the optional-add-on
  user contract)
- `setup_k3.py`, `test_setup_k3.py` and `test_model_source_pin.py` (`deltafin
  setup` and `deltafin setup-k3` own the immutable source pin, exact metadata
  allowlist, 96-shard inventory construction, full/stream capacity plan,
  default DSpark orchestration and transactional publication contracts)
- `test_warm_expert_cache.py` (superseded by native CLI, cache-warm,
  downloader and legacy-NPZ tests)
- `upgrade.py` (`deltafin upgrade` owns the clean-worktree, HTTPS-only
  fast-forward, model-data preservation, isolated native Git/Cargo execution,
  locked rebuild, build-profile replay and transitive loader-audit contract)
  together with its superseded Python-only test root

`native/deltafin/tests/release_policy.rs` enforces this boundary. It rejects
interpreted public commands, interpreted production-source extensions,
reappearance of retired roots, and any new top-level Python entrypoint that is
not explicitly classified as quarantined or development-only.
