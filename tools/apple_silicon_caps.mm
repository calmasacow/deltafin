// Read-only Metal capability probe used by tools/apple_silicon.py.
//
// This intentionally reports the device name only as provenance. Runtime
// decisions use supportsFamily:, feature properties, and resource limits.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            puts("{\"available\":false}");
            return 0;
        }

        NSMutableArray<NSString *> *families = [NSMutableArray array];
        for (NSInteger family = MTLGPUFamilyApple1;
             family <= MTLGPUFamilyApple10;
             ++family) {
            if ([device supportsFamily:(MTLGPUFamily)family]) {
                [families addObject:
                    [NSString stringWithFormat:@"apple%ld",
                                               (long)(family - 1000)]];
            }
        }
        const struct {
            MTLGPUFamily family;
            __unsafe_unretained NSString *name;
        } other_families[] = {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            // Keep the legacy family in the provenance snapshot even though
            // current SDKs prefer Mac2. This is reporting, not dispatch.
            {MTLGPUFamilyMac1, @"mac1"},
#pragma clang diagnostic pop
            {MTLGPUFamilyMac2, @"mac2"},
            {MTLGPUFamilyCommon1, @"common1"},
            {MTLGPUFamilyCommon2, @"common2"},
            {MTLGPUFamilyCommon3, @"common3"},
            {MTLGPUFamilyMetal3, @"metal3"},
            {MTLGPUFamilyMetal4, @"metal4"},
        };
        for (const auto &entry : other_families) {
            if ([device supportsFamily:entry.family]) {
                [families addObject:entry.name];
            }
        }

        NSMutableDictionary *features = [NSMutableDictionary dictionary];
        features[@"has_unified_memory"] = @(device.hasUnifiedMemory);
        features[@"bc_texture_compression"] =
            @(device.supportsBCTextureCompression);
        features[@"raytracing"] = @(device.supportsRaytracing);
        features[@"dynamic_libraries"] = @(device.supportsDynamicLibraries);
        features[@"function_pointers"] = @(device.supportsFunctionPointers);
        if (@available(macOS 26.4, *)) {
            features[@"placement_sparse"] =
                @(device.supportsPlacementSparse);
        }
        if ([(NSObject *)device
                respondsToSelector:@selector(supportsRaytracingFromRender)]) {
            features[@"raytracing_from_render"] =
                @(device.supportsRaytracingFromRender);
        }

        NSDictionary *result = @{
            @"available": @YES,
            @"name": device.name,
            @"registry_id": @(device.registryID),
            @"supported_families": families,
            @"features": features,
            @"recommended_max_working_set_bytes":
                @(device.recommendedMaxWorkingSetSize),
            @"max_buffer_length_bytes": @(device.maxBufferLength),
            @"max_threadgroup_memory_bytes":
                @(device.maxThreadgroupMemoryLength),
            @"max_transfer_rate_bytes_per_second": @(device.maxTransferRate),
            @"argument_buffers_tier": @(device.argumentBuffersSupport),
        };
        NSError *error = nil;
        NSData *json = [NSJSONSerialization dataWithJSONObject:result
                                                       options:0
                                                         error:&error];
        if (json == nil) {
            fprintf(stderr, "JSON serialization failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 2;
        }
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
    }
    return 0;
}
