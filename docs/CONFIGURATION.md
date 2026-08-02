# Configuration

Normal operation needs no environment overrides. The most useful controls are:

| Setting | Default | Meaning |
|---|---|---|
| `--device` / `K3_DEV` | `auto` | `mps`, `cuda`, `cuda:N` or `cpu`, with capability-gated auto selection |
| `--expert-backend` / `K3_MOE` | `auto` | `metal`, `cuda` or `cpu`, with capability-gated auto selection |
| `--spine` / `K3_SPINE` | `auto` | `auto` and `bf16` mean original weights; `int8` is explicit and non-weight-exact |
| `K3_EXPERT_SCALE4` | `auto` | `auto`, `off` or `require` for complete lossless scale4 sidecars |
| `K3_DSPARK` | `auto` | `auto`, `off` or force-qualified `on`; K3 verification is never bypassed |
| `K3_DSPARK_MAX_CONTEXT` | `8192` | bounded auxiliary draft-state context; full K3 continues above it |
| `K3_UAG_DRAFT` | `auto` | optional Qwen raw-completion policy: `auto`, `off` or `on` |
| `K3_TRACE` / `K3_TRACE_PATH` | `off` | native router trace mode and path; CLI flags are preferred |

The quality guard rejects fewer than 16 experts, non-fp32 target activations and approximation switches. Original BF16 remains the automatic resident authority. Optional paths must validate their device, ABI, shapes, memory and correctness before activation.
