#include "provider_route_mailbox.h"

#if !defined(__APPLE__)
#error "provider_route_mailbox.mm is an Apple-only Objective-C++ source"
#endif

#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#if !__has_feature(objc_arc)
#error "Deltafin's route-mailbox Metal bridge requires Objective-C ARC"
#endif

#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#include "deltafin_embedded_route_mailbox_metallib.h"
#else
#include "deltafin_embedded_route_mailbox_msl.h"
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

namespace deltafin::provider_internal {
namespace {

static_assert(offsetof(RouteMailboxT1, expert_ids) == 0);
static_assert(offsetof(RouteMailboxT1, weight_bits) == 128);
static_assert(sizeof(RouteMailboxT1) == 192);

struct MailboxState {
  std::mutex mutex;
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> mailbox = nil;
  id<MTLSharedEvent> event = nil;
  MTLSharedEventListener* listener = nil;
  std::uint64_t next_event = 1;
  bool rejected = false;
};

MailboxState& mailbox_state() {
  static MailboxState value;
  return value;
}

#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
dispatch_data_t copy_embedded_route_metallib() {
  void* owned = std::malloc(kDeltafinEmbeddedRouteMailboxMetallibBytes);
  if (owned == nullptr) return nullptr;
  std::memcpy(owned, kDeltafinEmbeddedRouteMailboxMetallib,
              kDeltafinEmbeddedRouteMailboxMetallibBytes);
  dispatch_data_t data = dispatch_data_create(
      owned, kDeltafinEmbeddedRouteMailboxMetallibBytes, nullptr,
      DISPATCH_DATA_DESTRUCTOR_FREE);
  if (data == nullptr) std::free(owned);
  return data;
}
#endif

id<MTLBuffer> tensor_buffer(const at::Tensor& tensor) {
  // This is the same reviewed MPS storage representation used by the existing
  // current-stream Metal bridges in this repository.
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

bool qualifies(const at::Tensor& expert_ids, const at::Tensor& weights) {
  return expert_ids.defined() && weights.defined() &&
         expert_ids.device().is_mps() && weights.device().is_mps() &&
         expert_ids.scalar_type() == at::kLong &&
         weights.scalar_type() == at::kFloat && expert_ids.is_contiguous() &&
         weights.is_contiguous() && !expert_ids.requires_grad() &&
         !weights.requires_grad() &&
         expert_ids.numel() == static_cast<std::int64_t>(kRouteMailboxTopK) &&
         weights.numel() == static_cast<std::int64_t>(kRouteMailboxTopK);
}

NSUInteger checked_byte_offset(const at::Tensor& tensor,
                               const NSUInteger bytes) {
  TORCH_CHECK(tensor.storage_offset() >= 0,
              "route mailbox tensor has a negative storage offset");
  const auto elements = static_cast<std::uint64_t>(tensor.storage_offset());
  const auto width = static_cast<std::uint64_t>(tensor.element_size());
  TORCH_CHECK(elements <= std::numeric_limits<std::uint64_t>::max() / width,
              "route mailbox tensor offset overflows uint64");
  const std::uint64_t raw = elements * width;
  TORCH_CHECK(raw <= std::numeric_limits<NSUInteger>::max(),
              "route mailbox tensor offset exceeds NSUInteger");
  id<MTLBuffer> buffer = tensor_buffer(tensor);
  TORCH_CHECK(buffer != nil, "route mailbox tensor has no MTLBuffer");
  const NSUInteger offset = static_cast<NSUInteger>(raw);
  TORCH_CHECK(offset <= buffer.length && bytes <= buffer.length - offset,
              "route mailbox tensor span exceeds its MTLBuffer");
  return offset;
}

void ensure_resources(MailboxState& state, id<MTLDevice> device) {
  if (state.device == device && state.pipeline != nil &&
      state.mailbox != nil && state.event != nil && state.listener != nil) {
    return;
  }

  state.device = device;
  state.pipeline = nil;
  state.mailbox = nil;
  state.event = nil;
  state.listener = nil;
  state.next_event = 1;

  NSError* error = nil;
#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
  dispatch_data_t data = copy_embedded_route_metallib();
  TORCH_CHECK(data != nullptr,
              "route mailbox embedded metallib wrapping failed");
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
  TORCH_CHECK(library != nil, "route mailbox metallib loading failed: ",
              error == nil ? "unknown"
                           : error.localizedDescription.UTF8String);
#else
  MTLCompileOptions* options = [MTLCompileOptions new];
  // The standalone development graph retains source compilation for its
  // shader gate. Cargo production builds define the precompiled branch above.
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString
                               stringWithUTF8String:
                                   kDeltafinEmbeddedRouteMailboxMsl]
                   options:options
                     error:&error];
  TORCH_CHECK(library != nil, "route mailbox Metal compilation failed: ",
              error == nil ? "unknown"
                           : error.localizedDescription.UTF8String);
#endif
  id<MTLFunction> function =
      [library newFunctionWithName:@"deltafin_route_mailbox_t1"];
  TORCH_CHECK(function != nil, "route mailbox Metal function is missing");
  state.pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  TORCH_CHECK(state.pipeline != nil,
              "route mailbox Metal pipeline creation failed: ",
              error == nil ? "unknown"
                           : error.localizedDescription.UTF8String);
  state.mailbox = [device
      newBufferWithLength:sizeof(RouteMailboxT1)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeDefaultCache];
  TORCH_CHECK(state.mailbox != nil,
              "route mailbox shared allocation failed");
  state.event = [device newSharedEvent];
  TORCH_CHECK(state.event != nil,
              "route mailbox shared event is unavailable");
  dispatch_queue_t listener_queue = dispatch_queue_create(
      "deltafin.route-mailbox", DISPATCH_QUEUE_SERIAL);
  state.listener = [[MTLSharedEventListener alloc]
      initWithDispatchQueue:listener_queue];
  TORCH_CHECK(state.listener != nil,
              "route mailbox event listener is unavailable");
}

void materialize(const at::Tensor& expert_ids, const at::Tensor& weights,
                 RouteMailboxT1& output) {
  TORCH_CHECK(qualifies(expert_ids, weights),
              "route mailbox inputs do not satisfy its T=1 contract");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "current MPS stream is unavailable");

  MailboxState& state = mailbox_state();
  std::lock_guard<std::mutex> guard(state.mutex);
  TORCH_CHECK(!state.rejected, "route mailbox was rejected by an earlier gate");
  id<MTLDevice> device = stream->device();
  TORCH_CHECK(device != nil, "current MPS device is unavailable");
  ensure_resources(state, device);

  constexpr NSUInteger kIdBytes =
      kRouteMailboxTopK * sizeof(std::int64_t);
  constexpr NSUInteger kWeightBytes =
      kRouteMailboxTopK * sizeof(std::uint32_t);
  const NSUInteger id_offset = checked_byte_offset(expert_ids, kIdBytes);
  const NSUInteger weight_offset = checked_byte_offset(weights, kWeightBytes);
  TORCH_CHECK(state.next_event != std::numeric_limits<std::uint64_t>::max(),
              "route mailbox event counter exhausted");
  const std::uint64_t event_value = state.next_event++;
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  [state.event notifyListener:state.listener
                     atValue:event_value
                       block:^(id<MTLSharedEvent>, std::uint64_t) {
                         dispatch_semaphore_signal(ready);
                       }];

  __block id<MTLCommandBuffer> root_command = nil;
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      TORCH_CHECK(encoder != nil,
                  "route mailbox current-stream encoder is unavailable");
      [encoder setComputePipelineState:state.pipeline];
      [encoder setBuffer:tensor_buffer(expert_ids)
                  offset:id_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights)
                  offset:weight_offset
                 atIndex:1];
      [encoder setBuffer:state.mailbox offset:0 atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(kRouteMailboxTopK, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kRouteMailboxTopK, 1, 1)];
      stream->endKernelCoalescing();
      MPSCommandBuffer* command = stream->commandBuffer();
      TORCH_CHECK(command != nil,
                  "route mailbox MPS command buffer is unavailable");
      root_command = command.rootCommandBuffer;
      TORCH_CHECK(root_command != nil,
                  "route mailbox root command buffer is unavailable");
      [root_command encodeSignalEvent:state.event value:event_value];
    }
  });

  // Commit precisely through the mailbox and let later MPS work begin on a
  // fresh command buffer.  Waiting on this event is narrower than a global MPS
  // synchronization and keeps both route tensors under one host boundary.
  stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  const long wait_result = dispatch_semaphore_wait(
      ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
  TORCH_CHECK(wait_result == 0, "route mailbox event timed out");
  TORCH_CHECK(root_command.error == nil, "route mailbox command failed: ",
              root_command.error.localizedDescription.UTF8String);
  const void* contents = state.mailbox.contents;
  TORCH_CHECK(contents != nullptr,
              "route mailbox shared storage is not CPU-visible");
  std::memcpy(&output, contents, sizeof(output));
}

}  // namespace

extern "C" int deltafin_route_mailbox_metallib_cycle_test_v1(int cycles) {
#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
  if (cycles <= 0 || cycles > 256) return -1;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return -2;
    for (int cycle = 0; cycle < cycles; ++cycle) {
      @autoreleasepool {
        dispatch_data_t data = copy_embedded_route_metallib();
        if (data == nullptr) return -3;
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithData:data error:&error];
        if (library == nil || error != nil) return -4;
      }
    }
  }
  return 0;
#else
  (void)cycles;
  return -5;
#endif
}

bool try_materialize_mps_route_t1(const at::Tensor& expert_ids,
                                  const at::Tensor& weights,
                                  RouteMailboxT1& output) noexcept {
  RouteMailboxT1 candidate;
  try {
    if (!qualifies(expert_ids, weights)) {
      return false;
    }
    materialize(expert_ids, weights, candidate);
  } catch (...) {
    // A private bridge must never make target routing unavailable.  Reject it
    // process-wide after any setup/runtime fault and let every caller use the
    // established LibTorch .to(CPU) path instead.
    try {
      MailboxState& state = mailbox_state();
      std::lock_guard<std::mutex> guard(state.mutex);
      state.rejected = true;
    } catch (...) {
      // Even failure bookkeeping is subordinate to preserving the fallback.
    }
    return false;
  }
  output = candidate;
  return true;
}

}  // namespace deltafin::provider_internal
