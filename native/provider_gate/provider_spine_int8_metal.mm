#if !defined(__APPLE__)
#error "provider_spine_int8_metal.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_SPINE_INT8_METAL_V1)
#error "int8 spine Metal bridge requires its explicit production capability"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "int8 spine Metal bridge requires an embedded precompiled metallib"
#endif

#include "provider_spine_int8_metal.h"

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#if !__has_feature(objc_arc)
#error "int8 spine Metal bridge requires Objective-C ARC"
#endif

#include "deltafin_embedded_spine_int8_metal_metallib.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr std::uint32_t kThreadsPerThreadgroup = 256;
constexpr std::uint32_t kThreadgroupColumns = 32;
constexpr std::uint32_t kThreadgroupRows =
    kThreadsPerThreadgroup / kThreadgroupColumns;

struct DequantDimsV1 {
  std::uint32_t rows;
  std::uint32_t columns;
  std::uint32_t elements;
  std::uint32_t reserved;
};

static_assert(sizeof(DequantDimsV1) == 16);

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
  if (!condition) fail(message);
}

dispatch_data_t copy_embedded_metallib() {
  void* owned = std::malloc(kDeltafinEmbeddedSpineInt8MetalMetallibBytes);
  if (owned == nullptr) return nullptr;
  std::memcpy(owned, kDeltafinEmbeddedSpineInt8MetalMetallib,
              kDeltafinEmbeddedSpineInt8MetalMetallibBytes);
  dispatch_data_t data = dispatch_data_create(
      owned, kDeltafinEmbeddedSpineInt8MetalMetallibBytes, nullptr,
      DISPATCH_DATA_DESTRUCTOR_FREE);
  if (data == nullptr) std::free(owned);
  return data;
}

struct MetalPipeline {
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> dequant = nil;
};

struct PipelineCache {
  std::mutex mutex;
  MetalPipeline pipeline;
};

PipelineCache& pipeline_cache() {
  static PipelineCache cache;
  return cache;
}

MetalPipeline pipeline_for(id<MTLDevice> device) {
  require(device != nil, "int8 spine Metal device is unavailable");
  PipelineCache& cache = pipeline_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (cache.pipeline.device != device || cache.pipeline.dequant == nil) {
    dispatch_data_t data = copy_embedded_metallib();
    require(data != nullptr, "wrap embedded int8 spine metallib failed");
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    require(library != nil,
            std::string("load embedded int8 spine metallib failed: ") +
                (error == nil ? "unknown"
                              : error.localizedDescription.UTF8String));
    id<MTLFunction> function =
        [library newFunctionWithName:@"deltafin_spine_int8_dequant_f32_v1"];
    require(function != nil,
            "embedded int8 spine metallib is missing its dequant function");
    id<MTLComputePipelineState> dequant =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(dequant != nil,
            std::string("create int8 spine Metal pipeline failed: ") +
                (error == nil ? "unknown"
                              : error.localizedDescription.UTF8String));
    require(dequant.maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
            "int8 spine Metal pipeline cannot admit 256 threads");
    cache.pipeline = MetalPipeline{device, library, dequant};
  }
  return cache.pipeline;
}

id<MTLBuffer> tensor_buffer(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

NSUInteger tensor_byte_offset(const at::Tensor& tensor,
                              const std::size_t required_bytes,
                              const char* name) {
  require(tensor.storage_offset() >= 0,
          std::string(name) + " has a negative storage offset");
  const std::uint64_t elements =
      static_cast<std::uint64_t>(tensor.storage_offset());
  const std::uint64_t width =
      static_cast<std::uint64_t>(tensor.element_size());
  require(elements <= std::numeric_limits<std::uint64_t>::max() / width,
          std::string(name) + " storage offset overflows uint64");
  const std::uint64_t raw = elements * width;
  require(raw <= std::numeric_limits<NSUInteger>::max(),
          std::string(name) + " storage offset exceeds NSUInteger");
  id<MTLBuffer> buffer = tensor_buffer(tensor);
  require(buffer != nil, std::string(name) + " has no MTLBuffer");
  const NSUInteger offset = static_cast<NSUInteger>(raw);
  require(offset <= buffer.length &&
              required_bytes <= buffer.length - offset,
          std::string(name) + " span exceeds its MTLBuffer");
  return offset;
}

void require_tensor(const at::Tensor& tensor, const at::Device& device,
                    const at::ScalarType dtype,
                    const at::IntArrayRef shape, const char* name) {
  require(tensor.defined() && tensor.device() == device &&
              tensor.scalar_type() == dtype && tensor.is_contiguous() &&
              tensor.sizes() == shape,
          std::string("int8 spine ") + name +
              " has an invalid device/dtype/shape/layout");
}

}  // namespace

SpineInt8MetalCapabilities spine_int8_metal_capabilities_v1() {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  const MetalPipeline pipeline = pipeline_for(stream->device());
  require(pipeline.dequant != nil,
          "int8 spine Metal pipeline is incomplete");
  return SpineInt8MetalCapabilities{
      .abi_version = kSpineInt8MetalAbiV1,
      .flags = kSpineInt8MetalRequiredCapabilitiesV1,
      .threads_per_threadgroup = kThreadsPerThreadgroup,
      .reserved = 0,
  };
}

void spine_int8_metal_dequant_f32(const at::Tensor& destination,
                                 const at::Tensor& quantized,
                                 const at::Tensor& row_scales) {
  require(quantized.defined() && quantized.dim() == 2,
          "int8 spine quantized matrix must have rank two");
  const std::int64_t rows = quantized.size(0);
  const std::int64_t columns = quantized.size(1);
  require(rows > 0 && columns > 0 &&
              rows <= std::numeric_limits<std::uint32_t>::max() &&
              columns <= std::numeric_limits<std::uint32_t>::max() &&
              rows <= std::numeric_limits<std::uint32_t>::max() / columns,
          "int8 spine matrix dimensions exceed the Metal ABI");
  const at::Device device = quantized.device();
  require(device.is_mps(), "int8 spine dequant requires MPS tensors");
  require_tensor(quantized, device, at::kChar, {rows, columns},
                 "quantized matrix");
  require_tensor(row_scales, device, at::kFloat, {rows}, "row scales");
  require_tensor(destination, device, at::kFloat, {rows, columns},
                 "destination");

  const std::uint64_t elements64 =
      static_cast<std::uint64_t>(rows) * columns;
  require(elements64 <= std::numeric_limits<std::uint32_t>::max(),
          "int8 spine matrix element count exceeds the Metal grid ABI");
  const std::uint32_t elements = static_cast<std::uint32_t>(elements64);
  const std::size_t quantized_bytes = static_cast<std::size_t>(elements);
  const std::size_t scale_bytes =
      static_cast<std::size_t>(rows) * sizeof(float);
  const std::size_t destination_bytes =
      static_cast<std::size_t>(elements) * sizeof(float);
  const NSUInteger quantized_offset = tensor_byte_offset(
      quantized, quantized_bytes, "int8 spine quantized matrix");
  const NSUInteger scale_offset =
      tensor_byte_offset(row_scales, scale_bytes, "int8 spine row scales");
  const NSUInteger destination_offset = tensor_byte_offset(
      destination, destination_bytes, "int8 spine destination");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr && stream->device() == tensor_buffer(quantized).device,
          "int8 spine tensors belong to another MPS device");
  const MetalPipeline pipeline = pipeline_for(stream->device());
  const DequantDimsV1 dims{
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(columns), elements, 0};
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      require(encoder != nil,
              "current MPS stream has no int8 spine encoder");
      [encoder setComputePipelineState:pipeline.dequant];
      [encoder setBuffer:tensor_buffer(quantized)
                  offset:quantized_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(row_scales)
                  offset:scale_offset
                 atIndex:1];
      [encoder setBuffer:tensor_buffer(destination)
                  offset:destination_offset
                 atIndex:2];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
      // A 2D grid gives the row directly. The earlier scalar prototype paid
      // one runtime integer division per weight to recover index/columns,
      // which is unacceptable for layer 0's 1.17 billion-element template.
      [encoder dispatchThreads:
                   MTLSizeMake(static_cast<NSUInteger>(columns),
                               static_cast<NSUInteger>(rows), 1)
          threadsPerThreadgroup:
              MTLSizeMake(kThreadgroupColumns, kThreadgroupRows, 1)];
    }
  });
}

SpineInt8MetalCanaryReport spine_int8_metal_canary_v1() {
  constexpr std::int64_t kRows = 257;
  constexpr std::int64_t kColumns = 263;
  constexpr std::int64_t kQuantizedOffset = 37;
  constexpr std::int64_t kScaleOffset = 11;
  constexpr std::int64_t kDestinationOffset = 67;
  constexpr std::int64_t kElements = kRows * kColumns;

  std::vector<std::int8_t> quantized_values(kElements);
  std::vector<float> scale_values(kRows);
  std::vector<float> expected(kElements);
  for (std::int64_t index = 0; index < kElements; ++index) {
    quantized_values[index] = static_cast<std::int8_t>(
        static_cast<int>((index * 73 + 19) % 256) - 128);
  }
  for (std::int64_t row = 0; row < kRows; ++row) {
    const std::int32_t centered =
        static_cast<std::int32_t>((row * 41 + 7) % 193) - 96;
    scale_values[row] = static_cast<float>(centered) / 256.0F;
    for (std::int64_t column = 0; column < kColumns; ++column) {
      const std::int64_t index = row * kColumns + column;
      expected[index] =
          static_cast<float>(quantized_values[index]) * scale_values[row];
    }
  }

  at::Tensor quantized_storage = at::zeros(
      {kQuantizedOffset + kElements + 13},
      at::TensorOptions().dtype(at::kChar).device(at::kMPS));
  at::Tensor scale_storage = at::zeros(
      {kScaleOffset + kRows + 13},
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  at::Tensor destination_storage = at::full(
      {kDestinationOffset + kElements + 13}, -123.0F,
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  at::Tensor quantized = quantized_storage.narrow(
      0, kQuantizedOffset, kElements).view({kRows, kColumns});
  at::Tensor scales = scale_storage.narrow(0, kScaleOffset, kRows);
  at::Tensor destination = destination_storage.narrow(
      0, kDestinationOffset, kElements).view({kRows, kColumns});
  const at::Tensor quantized_cpu = at::from_blob(
      quantized_values.data(), {kRows, kColumns},
      at::TensorOptions().dtype(at::kChar).device(at::kCPU));
  const at::Tensor scales_cpu = at::from_blob(
      scale_values.data(), {kRows},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  quantized.copy_(quantized_cpu, false);
  scales.copy_(scales_cpu, false);

  spine_int8_metal_dequant_f32(destination, quantized, scales);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr,
          "current MPS stream disappeared during int8 spine canary");
  stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  const at::Tensor output = destination.to(at::kCPU).contiguous();
  const float* values = output.const_data_ptr<float>();

  SpineInt8MetalCanaryReport report{
      .rows = static_cast<std::uint32_t>(kRows),
      .columns = static_cast<std::uint32_t>(kColumns),
      .compared_elements = static_cast<std::uint32_t>(kElements),
      .source_offset_elements =
          static_cast<std::uint32_t>(kQuantizedOffset),
      .scale_offset_elements = static_cast<std::uint32_t>(kScaleOffset),
      .destination_offset_elements =
          static_cast<std::uint32_t>(kDestinationOffset),
  };
  for (std::int64_t index = 0; index < kElements; ++index) {
    if (!std::isfinite(values[index]) || !std::isfinite(expected[index])) {
      ++report.nonfinite;
    } else if (std::bit_cast<std::uint32_t>(values[index]) ==
               std::bit_cast<std::uint32_t>(expected[index])) {
      ++report.equal_bits;
    }
  }
  return report;
}

}  // namespace deltafin::provider_internal
