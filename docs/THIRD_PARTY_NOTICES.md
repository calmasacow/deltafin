# Third-party provenance and notices

Deltafin distinguishes immutable upstream data, linked native dependencies,
source-only design references and Deltafin-owned implementations. This file is
an attribution record; it does not replace the license distributed by each
upstream project or model.

| Component | Material used | Deltafin-owned work | Install and execution boundary | License / credit |
|---|---|---|---|---|
| [Moonshot Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | Model weights, configuration, tokenizer data and license pinned at `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721` | Native model parser, tensor plan, target implementation, storage and acceleration | `deltafin setup` downloads only an allowlisted inert metadata set and canonical tensor ranges. Upstream model source is not imported or executed. Model files remain outside Git. | [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/LICENSE) |
| [Inferact Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) | Unmodified, data-only draft checkpoint pinned at `cf6b8244620e7ea4b0651d214f28e89eac75bed6` | Rust checkpoint admission and scheduling plus a Deltafin C++/LibTorch model provider, target-row capture, verification and state transactions | Native setup accepts exactly five pinned files, validates sizes, SHA-256 and the full Safetensors schema, and never loads remote code. The checkpoint is stored in ignored `k3-draft-dspark/`. | [Kimi K3 License](https://huggingface.co/Inferact/Kimi-K3-DSpark/blob/cf6b8244620e7ea4b0651d214f28e89eac75bed6/LICENSE); [Inferact](https://huggingface.co/Inferact) credited |
| [Qwen3 0.6B Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) and [Qwen3 1.7B Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | Optional unmodified, data-only proposal checkpoints pinned at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` and `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | Strict native checkpoint parser, Rust proposal controller, cross-tokenizer verification and C++/LibTorch provider | `deltafin setup-qwen` accepts fixed eight-file allowlists for each model. No upstream executable source is downloaded or loaded. | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/LICENSE), Qwen team |
| [PyTorch / LibTorch](https://github.com/pytorch/pytorch) 2.13.0 | Compiled C++ tensor providers and headers | Direct provider ABI, target sequencing, lifetime gates and custom specialized kernels around ATen | The normal native build authenticates one pinned official CPU artifact and consumes its native C++ layout without evaluating bundled build metadata; `libtorch_python` is excluded. CUDA builds use an operator-supplied audited CUDA LibTorch tree and matching toolkit. | [BSD-style PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [curl-rust](https://github.com/alexcrichton/curl-rust) `curl-sys` 0.4.90+curl-8.21.0 | Upstream FFI declarations and license, preserved byte-for-byte | Deltafin-maintained Rust-only system-library selection, ELF validation, link setup and runtime capability admission | The local fork is distributed as `native/deltafin-curl-sys-direct/`. It has no build dependencies and starts no discovery, shell, package-manager or source-build helper. macOS links the system libcurl directly; supported Linux targets accept only a bounded, root-owned `libcurl.so.4` that passes static ABI checks, followed by runtime libcurl 7.28+/TLS/HTTPS gates. | [MIT](https://github.com/alexcrichton/curl-rust/blob/main/LICENSE), Alex Crichton and curl-rust contributors |
| [GigaToken](https://github.com/marcelroed/gigatoken) 0.10.0 | Source/interface study of stable-order native batch tokenization | Independent exact K3 rank-file tokenizer and automatic large-history segment fan-out in Rust | No GigaToken package, wheel or source is installed, copied, linked or executed by the production runtime. | [MIT](https://github.com/marcelroed/gigatoken/blob/main/LICENSE), Copyright © 2026 [Marcel Rød](https://github.com/marcelroed) |
| [vLLM](https://github.com/vllm-project/vllm) | Source-only study of K3 recurrent/attention cache coordination at commit `0f17394564fa2fccd332cf63321314884c15ee37` | Narrow one-owner provider branch transaction and exact strict-extension reuse | No vLLM package or source is installed, copied, built or executed by Deltafin. | [Apache-2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE); no vLLM code is distributed here |
| [TorchSpec](https://github.com/lightseekorg/TorchSpec) | DSpark's published training framework and architecture context | None claimed | Source/design reference only; not bundled, installed or executed. | Upstream license and authors |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Kimi Delta Attention semantics and reference behavior | Deltafin provider implementation and native KDA state lifecycle | Production inference uses the compiled Deltafin provider. Historical reference material under `tools/fla/` retains attribution in its headers and the root license. | [MIT](https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE), Copyright © 2023–2026 Songlin Yang, Yu Zhang and Zhiyuan Li |
| [colibri](https://github.com/JustVugg/colibri) | Source-only study of MoE streaming, router lookahead and platform scheduling | Bounded authenticated readers and Deltafin's own exact scheduling-only PILOT path | Not installed as a runtime dependency. | [Apache-2.0](https://github.com/JustVugg/colibri/blob/main/LICENSE), JustVugg |
| [ds4 / DwarfStar](https://github.com/antirez/ds4) | Source-only study of expert streaming, ownership, eviction and correctness measurement | Deltafin's own storage formats, provider ABI and cache policies | Not installed as a runtime dependency. | [MIT](https://github.com/antirez/ds4/blob/main/LICENSE), Salvatore Sanfilippo |

## Compiled Rust dependencies

The production binary uses the exact versions and checksums recorded in
`Cargo.lock`. Direct dependencies are:

- `base64` 0.22.1 for strict K3 rank-file decoding;
- `curl` 0.4.50, Deltafin's audited local `curl-sys` fork and the qualified
  platform libcurl stack for bounded authenticated HTTPS;
- `half` 2.7.1 for explicit float-format handling;
- `libc` 0.2.189 for checked operating-system boundaries;
- `regex` 1.13.0 for K3's fixed Unicode pre-tokenization expression;
- `rustc-hash` 2.1.3 for bounded in-memory tokenizer tables;
- `serde` 1.0.229 and `serde_json` 1.0.145 for strict data contracts;
- `tiny_http` 0.12.0 for the in-process OpenAI-compatible server;
- `tokenizers` 0.22.2 for the optional Qwen tokenizer data format.

Their complete transitive dependency and checksum record is the locked
workspace manifest. The K3 tokenizer-specific security review is in
`native/deltafin/TOKENIZER_AUDIT.md`.

## What a user receives

Cloning the repository provides Deltafin's Rust source, C++ provider and
specialized C/Metal/CUDA sources. Building produces one native executable.
Normal setup downloads pinned model data into ignored directories; it does not
download an alternate runtime or execute repository-defined model code.

Installing an upstream package or checkpoint by itself does not provide
Deltafin's scheduling, verification, cache transactions or safety gates.
Conversely, the large local model directories do not belong in Git: the native
reproducible installer and Deltafin-owned runtime are the distributed project.

## Policy for future upstream modifications

If Deltafin ever modifies upstream source instead of independently
implementing or wrapping a public contract, it must be represented explicitly
as either:

1. a clearly named maintained fork pinned to an exact commit; or
2. `third_party/<project>/` containing the upstream URL/revision, intact
   license and notice files, a local-change record, and reproducible build and
   verification steps.

An unchanged downloaded model directory is not a fork. A source-only design
reference is not a bundled dependency. Those boundaries must remain clear in
code, setup and documentation.
