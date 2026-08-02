#include "provider_abi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

static_assert(sizeof(DeltafinProviderTargetSequencePlanExpertsRequestV1) ==
              240);
static_assert(sizeof(DeltafinProviderTargetSequencePlanExpertsReportV1) ==
              208);
static_assert(
    sizeof(DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1) ==
    240);
static_assert(offsetof(DeltafinProviderTargetSequencePlanExpertsRequestV1,
                       expert_ids) == 60);
static_assert(offsetof(DeltafinProviderTargetSequencePlanExpertsReportV1,
                       missing_experts) == 56);

template <typename Function>
void require_failure(Function&& function, const std::string_view expected) {
  std::array<char, 1024> error{};
  if (function(error.data(), error.size()) == 0) {
    throw std::runtime_error(
        "expected provider CUDA-plan ABI call to fail closed");
  }
  if (std::string_view(error.data()).find(expected) == std::string_view::npos) {
    throw std::runtime_error(
        std::string("unexpected provider CUDA-plan ABI failure: ") +
        error.data());
  }
}

DeltafinProviderSessionHandleV1 create_cpu_session() {
  DeltafinProviderSessionRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.requested_device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  request.max_route_positions = 1;
  DeltafinProviderSessionReportV1 report{};
  report.struct_size = sizeof(report);
  std::array<char, 1024> error{};
  if (deltafin_provider_session_create_v1(
          &request, &report, error.data(), error.size()) != 0) {
    throw std::runtime_error(
        std::string("could not create CUDA-plan test session: ") +
        error.data());
  }
  if (report.session == 0 ||
      report.selected_device != DELTAFIN_PROVIDER_DEVICE_CPU_V1) {
    throw std::runtime_error(
        "CUDA-plan test session did not select exact CPU");
  }
  return report.session;
}

void destroy_session(const DeltafinProviderSessionHandleV1 session) {
  DeltafinProviderResourceRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session;
  std::array<char, 1024> error{};
  if (deltafin_provider_session_destroy_v1(
          &request, error.data(), error.size()) != 0) {
    throw std::runtime_error(
        std::string("could not destroy CUDA-plan test session: ") +
        error.data());
  }
}

DeltafinProviderTargetSequencePlanExpertsRequestV1 valid_plan_request() {
  DeltafinProviderTargetSequencePlanExpertsRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.sequence = 1;
  request.spine_generation = 2;
  request.layer_index = 1;
  request.row_count = 1;
  request.expert_backend = DELTAFIN_PROVIDER_TARGET_EXPERT_AUTO_V1;
  request.cpu_threads = 1;
  request.expert_count = 1;
  request.expert_ids[0] = 7;
  return request;
}

void portable_request_gate_test() {
  {
    auto request = valid_plan_request();
    request.expert_backend = DELTAFIN_PROVIDER_TARGET_EXPERT_CPU_V1;
    DeltafinProviderTargetSequencePlanExpertsReportV1 report{};
    report.struct_size = sizeof(report);
    require_failure(
        [&](char* error, const std::size_t capacity) {
          return deltafin_provider_target_sequence_plan_experts_v1(
              &request, &report, error, capacity);
        },
        "invalid bounds/backend");
  }
  {
    auto request = valid_plan_request();
    request.expert_ids[1] = 8;
    DeltafinProviderTargetSequencePlanExpertsReportV1 report{};
    report.struct_size = sizeof(report);
    require_failure(
        [&](char* error, const std::size_t capacity) {
          return deltafin_provider_target_sequence_plan_experts_v1(
              &request, &report, error, capacity);
        },
        "unused expert-plan ID slots");
  }
}

void portable_session_gate_test() {
  const DeltafinProviderSessionHandleV1 session = create_cpu_session();
  try {
    auto plan = valid_plan_request();
    plan.session = session;
    DeltafinProviderTargetSequencePlanExpertsReportV1 plan_report{};
    plan_report.struct_size = sizeof(plan_report);
    require_failure(
        [&](char* error, const std::size_t capacity) {
          return deltafin_provider_target_sequence_plan_experts_v1(
              &plan, &plan_report, error, capacity);
        },
        "requires a selected CUDA provider");

    // A zero-miss/all-hit finish has canonical null/zero storage. It reaches
    // handle authentication rather than being rejected as a malformed byte
    // request; no physical CUDA execution is claimed by this portable test.
    DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1 finish{};
    finish.struct_size = sizeof(finish);
    finish.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    finish.session = session;
    finish.sequence = 1;
    finish.plan = 999;
    finish.spine_generation = 2;
    finish.layer_index = 1;
    finish.row_count = 1;
    DeltafinProviderTargetSequenceFinishExpertsReportV1 finish_report{};
    finish_report.struct_size = sizeof(finish_report);
    require_failure(
        [&](char* error, const std::size_t capacity) {
          return deltafin_provider_target_sequence_finish_planned_experts_v1(
              &finish, &finish_report, error, capacity);
        },
        "stale or unknown");

    std::uint8_t noncanonical = 0;
    finish.expert_major_bytes = &noncanonical;
    require_failure(
        [&](char* error, const std::size_t capacity) {
          return deltafin_provider_target_sequence_finish_planned_experts_v1(
              &finish, &finish_report, error, capacity);
        },
        "invalid bounds/pointer");

    // Releasing a plan never asks for a live target sequence. The unknown
    // handle is diagnosed directly, proving that teardown has no sequence
    // precondition on this portable path.
    DeltafinProviderResourceRequestV1 release{};
    release.struct_size = sizeof(release);
    release.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    release.session = session;
    release.resource = 999;
    require_failure(
        [&](char* error, const std::size_t capacity) {
          return deltafin_provider_moe_plan_release_v1(
              &release, error, capacity);
        },
        "stale or unknown");
  } catch (...) {
    destroy_session(session);
    throw;
  }
  destroy_session(session);
}

}  // namespace

int main() {
  try {
    portable_request_gate_test();
    portable_session_gate_test();
    std::cout << "provider_cuda_plan_abi.portable=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_cuda_plan_abi_test failed: " << error.what()
              << '\n';
    return 1;
  }
}
