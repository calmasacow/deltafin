# Supported platforms

| Host | Resident target and attention | Routed experts |
|---|---|---|
| Apple Silicon macOS | MPS | Metal, with native CPU fallback |
| NVIDIA Linux x86-64/aarch64 | CUDA | qualified native CUDA MXFP4, with native CPU fallback |
| Linux x86-64 | CPU | SSSE3/AVX/FMA3 compatibility kernel, with runtime-selected AVX2 acceleration |
| Linux aarch64 | CPU | native NEON MXFP4 |

Platform support describes implemented selection and safety gates, not equal benchmark coverage. Apple Silicon has maintainer-run full-model evidence; a DGX Spark community run established Linux/CUDA viability; additional physical CUDA and native Linux performance reports remain welcome. The compiled CPU fallback stays available when a GPU-specific expert provider does not qualify. Windows is not currently supported; it is a future portability target, not an implied capability of the Linux or macOS build.
