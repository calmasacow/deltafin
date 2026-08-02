#include "../../tools/metal_moe_abi.h"

#include <cstring>
#include <iostream>

extern "C" {
int k3_metal_init(const char* source_selector);
const char* k3_metal_last_error(void);
uint32_t k3_metal_descriptor_abi_version(void);
uint64_t k3_metal_layout_capabilities(void);
}

int main(int argc, char** argv) {
  const char* selector = K3_METAL_MOE_EMBEDDED_SOURCE_V1;
  const char* mode = "embedded-v1";
  if (argc == 3 && std::strcmp(argv[1], "--source") == 0 &&
      argv[2][0] != '\0') {
    selector = argv[2];
    mode = "explicit-development-path";
  } else if (argc != 1) {
    std::cerr << "usage: deltafin-provider-metal-source-test [--source PATH]\n";
    return 2;
  }

  if (k3_metal_init(selector) != 0) {
    const char* detail = k3_metal_last_error();
    std::cerr << "provider_metal_source=FAIL: "
              << (detail == nullptr ? "no detail" : detail) << '\n';
    return 1;
  }
  const uint64_t expected_layouts = K3_CAP_RAW_V1 | K3_CAP_SCALE4_V2;
  if (k3_metal_descriptor_abi_version() != K3_DESCRIPTOR_ABI_V1 ||
      (k3_metal_layout_capabilities() & expected_layouts) != expected_layouts) {
    std::cerr << "provider_metal_source=FAIL: embedded descriptor pipelines "
                 "are incomplete\n";
    return 1;
  }
  std::cout << "provider_metal_source=PASS\n"
            << "provider_metal_source.mode=" << mode << '\n';
  return 0;
}
