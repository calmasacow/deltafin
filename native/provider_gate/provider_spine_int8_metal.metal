#include <metal_stdlib>
using namespace metal;

struct DeltafinSpineInt8DequantDimsV1 {
  uint rows;
  uint columns;
  uint elements;
  uint reserved;
};

/*
 * This is the compiled form of the live runtime's proven expression. The
 * uploaded scale is already the exact fp32 expansion of its checkpoint fp16
 * value, so each output performs one fp32 conversion and one fp32 multiply.
 */
kernel void deltafin_spine_int8_dequant_f32_v1(
    device const char* quantized [[buffer(0)]],
    device const float* row_scales [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant DeltafinSpineInt8DequantDimsV1& dims [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x < dims.columns && position.y < dims.rows) {
    const uint index = position.y * dims.columns + position.x;
    destination[index] = float(quantized[index]) * row_scales[position.y];
  }
}
