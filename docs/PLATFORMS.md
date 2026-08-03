# Supported platforms

| Host | Resident target and attention | Routed experts |
|---|---|---|
| Apple Silicon macOS | MPS | Metal, with native CPU fallback |
| NVIDIA Linux x86-64/aarch64 | CUDA | qualified native CUDA MXFP4, with native CPU fallback |
| AMD Linux x86-64 (ROCm/HIP) | HIP, reported as CUDA devices | native CPU MXFP4 only; device MXFP4 kernels are NVIDIA-only |
| Linux x86-64 | CPU | SSSE3/AVX/FMA3 compatibility kernel, with runtime-selected AVX2 acceleration |
| Linux aarch64 | CPU | native NEON MXFP4 |

ROCm builds of PyTorch keep the `torch_cuda`/`c10_cuda` library names and the `at::kCUDA` device type, so Deltafin identifies them by their HIP ELF dependency and reports them as CUDA devices. Resident work and the int8 spine run on the GPU; routed experts stay on the exact CPU MXFP4 path, because the device MXFP4 and original-BF16 kernels reduce with a fixed 32-lane warp contract that is not valid on 64-lane AMD wavefronts. `DELTAFIN_CUDA_MOE=ON` therefore rejects a ROCm root rather than building kernels that would silently misreduce. No maintainer or community run has been recorded on AMD hardware yet.

Platform support describes implemented selection and safety gates, not equal benchmark coverage. Apple Silicon has maintainer-run full-model evidence; a DGX Spark community run established Linux/CUDA viability; additional physical CUDA and native Linux performance reports remain welcome. The compiled CPU fallback stays available when a GPU-specific expert provider does not qualify. Windows is not currently supported; it is a future portability target, not an implied capability of the Linux or macOS build.
