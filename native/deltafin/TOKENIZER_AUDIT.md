# Native K3 tokenizer audit

This records the boundary reviewed before Deltafin replaced Python ownership of
raw K3 tokenization and XTML chat rendering. It is an implementation audit, not
a claim that tokenization materially changes target-model generation speed: the
large benefit is removing Python from the eventual native runtime and avoiding
repeated orchestration/allocation overhead in long-lived server use.

## Authoritative local behavior

The implementation was derived from a complete review of the installed,
data-only tokenizer contract:

- `k3-meta/tokenization_kimi.py`: the eight-way Unicode split expression,
  400,000-character outer slices, 25,000-character homogeneous-run guard,
  tiktoken rank-priority BPE, special-token policy, and lossy UTF-8 decode;
- `k3-meta/encoding_k3.py`: XTML tags, per-segment trust labels, tool argument
  normalization, tool-result ordering, images, thinking effort, tool choice,
  and response-format messages;
- `k3-meta/tiktoken.model`: 163,584 base64 byte tokens with contiguous ranks;
  and
- `k3-meta/tokenizer_config.json`: 256 reserved IDs and their K3 spellings.

No external source was downloaded or executed for this port. The existing
local Python/tiktoken implementation was used only to produce fixed test-gold
IDs and renderings after its source and inputs were reviewed.

## Dependency review

All direct versions are exact in `native/deltafin/Cargo.toml`, all transitive
versions/checksums are fixed by `Cargo.lock`, and the first build was completed
with Cargo offline.

| Crate | Reviewed use | Build/code boundary |
|---|---|---|
| `base64` 0.22.1 | Strict RFC 4648 standard decoding of inert vocabulary lines | Pure Rust; no build script |
| `regex` 1.13.0 | One compile-time-constant K3 Unicode expression | Pure Rust, no build script, finite-automata matching with linear-time guarantee |
| `rustc-hash` 2.1.3 | Fast lookup in the bounded, locally installed rank table | Pure Rust; no build script; never used for security or integrity decisions |
| `serde_json` 1.0.145 | Bounded tokenizer config and chat JSON | Its 29-line build script only selects 32/64-bit arithmetic from Cargo target variables |
| `indexmap` 2.14.0 | `serde_json/preserve_order`, required because tool argument order changes exact token IDs | Pure Rust; no build script |

The regex path resolves to `regex-automata` 0.4.16, `regex-syntax` 0.8.11,
`aho-corasick` 1.1.4, and `memchr` 2.8.3. Ordered JSON resolves to `hashbrown`
0.17.1 and `equivalent` 1.0.2. Their cached manifests/source were inspected;
none has a build script. No dependency in this native K3 tokenizer path loads
native plugins, starts a process, accesses the network, or crosses an FFI
boundary.

### Optional Qwen assistant tokenizer

The optional Qwen raw-completion assistant uses the separately pinned
`tokenizers` 0.22.2 crate. Its default features remain disabled: Deltafin does
not build the unused progress UI or the `esaxx-rs` C++ trainer. Oniguruma is the
only enabled tokenizer feature, and its vendored C build is included explicitly
in the Cargo build-child audit.

That choice is measured rather than assumed. On 2026-08-01, two minimal release
binaries were built from the same Rust source and profile; their only difference
was `tokenizers`' `onig` versus `fancy-regex` feature. Both loaded the installed
`k3-draft-qwen3-0.6b-base/tokenizer.json`. A binary record containing every
input byte, token ID, and decoded byte was identical across 12 raw, chat, tool,
code, Unicode, whitespace, and growing-context cases, including a 268,749-byte
history. Both records had SHA-256
`76c726d6acea8a9600b0548c8fcbf14b763751f0df548e5b7bceb2bf9ae6d801`.
The installed 1.7B assistant carries the byte-identical 7,031,645-byte
tokenizer (SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`),
so the same result covers both optional Qwen variants.

Median warmed single-thread encode times nevertheless favored Oniguruma:

| Input | Bytes / IDs | Oniguruma | `fancy-regex` | Pure-Rust change |
|---|---:|---:|---:|---:|
| One-token proposal, ` Paris` | 6 / 1 | 2.786 us | 2.845 us | +2.1% |
| Twelve-token proposal | 45 / 12 | 12.805 us | 14.115 us | +10.2% |
| Short raw prompt | 24 / 5 | 7.374 us | 8.262 us | +12.0% |
| Code sample | 229 / 91 | 67.765 us | 80.668 us | +19.0% |
| Qwen chat text | 267 / 66 | 61.270 us | 70.825 us | +15.6% |
| Growing history | 6,891 / 1,540 | 1.337 ms | 1.657 ms | +23.9% |
| Growing history | 68,910 / 15,400 | 14.123 ms | 16.351 ms | +15.8% |
| Growing history | 268,749 / 60,060 | 59.810 ms | 66.275 ms | +10.8% |

Decode speed and one-time load time were effectively equal. Qwen re-encodes
the canonical target history for a proposal, so retaining the faster backend
matters most as a conversation grows. Deltafin therefore keeps Oniguruma while
turning off the unrelated default features; selecting a pure-Rust regex engine
solely to make the dependency graph look smaller would slow the live path.

## Safety and exactness controls

- Vocabulary/config files have hard byte limits and are parsed as inert data.
- Ranks must be unique, contiguous, ordered, and exactly 163,584 entries; all
  256 single-byte values must exist.
- The decoder must contain exactly 163,840 IDs, with K3's structural, media,
  BOS/EOS/UNK/PAD spellings at their audited IDs.
- Special tokens are ASCII, unique, and prefix-unambiguous. Ordinary user/tool
  segments never enable them.
- The original negative-lookahead whitespace semantics are implemented as an
  explicit boundary rule, allowing the rest of the expression to remain
  non-backtracking.
- BPE uses a linked token chain and invalidation-aware priority queue. Its key
  is `(rank, byte offset)`, preserving tiktoken's lowest-rank then leftmost tie
  rule without quadratic whole-vector rescans.
- Appending token IDs is transactional on error. Decode validates every ID
  before indexing and exposes raw bytes for correct incremental UTF-8 output.
- Chat JSON preserves object insertion order, while only the fields that the
  former reference contract explicitly deep-sorted are sorted.

## Parity gates

The Rust test corpus covers both ordinary and special-enabled encoding,
multilingual scripts, combining marks, emoji/ZWJ sequences, contractions,
numbers, CR/LF and uncommon Unicode whitespace, decode, injection-like literal
markers, all long-input safety boundaries, and 43,904 combinatorial boundary
cases. XTML fixed gold covers simple chat, multimodal/non-thinking chat, tool
declarations and arguments, invalid JSON blocks, reordered tool results,
thinking/tool-choice messages, and deep-sorted response schemas.

The intentionally adversarial 25,001-emoji test also guards the performance
shape: the initial exact vector-rescan prototype took about 3.37 seconds on the
local M1 Max, while the final priority-queue implementation took about 6.6 ms
and emitted the same 50,002 IDs. This is a tokenizer-kernel microcheck, not an
end-to-end tokens-per-second benchmark.
