#include "provider_abi.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::array<std::uint64_t, 2> shape_for(const std::uint32_t slot,
                                       std::uint32_t& rank) {
  rank = 2;
  switch (slot) {
    case 1: return {8, 40};
    case 2:
    case 3: rank = 1; return {8, 0};
    case 4:
    case 5: return {32, 4};
    case 6: return {1, 12};
    case 7: rank = 1; return {1, 0};
    default: break;
  }
  switch ((slot - 8) % 12) {
    case 0:
    case 1: rank = 1; return {8, 0};
    case 2: return {4, 8};
    case 3: rank = 1; return {4, 0};
    case 4: return {12, 4};
    case 5: return {8, 8};
    case 6: rank = 1; return {4, 0};
    case 7: return {8, 4};
    case 8: return {8, 4};
    case 9:
    case 10: return {12, 8};
    case 11: return {8, 12};
    default: throw std::logic_error("unreachable DSpark ABI slot");
  }
}

std::uint64_t elements(const std::array<std::uint64_t, 2>& shape,
                       const std::uint32_t rank) {
  return rank == 1 ? shape[0] : shape[0] * shape[1];
}

void require_ok(const int32_t status, const char* error,
                const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) + ": " + error);
  }
}

DeltafinProviderResourceRequestV1 resource(const std::uint64_t session,
                                           const std::uint64_t handle) {
  DeltafinProviderResourceRequestV1 request = {};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session;
  request.resource = handle;
  return request;
}

}  // namespace

int main() {
  try {
    char error[2048] = {};
    DeltafinProviderSessionRequestV1 session_request = {};
    session_request.struct_size = sizeof(session_request);
    session_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    session_request.requested_device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
    session_request.max_route_positions = 64;
    DeltafinProviderSessionReportV1 session_report = {};
    session_report.struct_size = sizeof(session_report);
    require_ok(deltafin_provider_session_create_v1(
                   &session_request, &session_report, error, sizeof(error)),
               error, "create session");

    std::array<std::vector<std::uint16_t>,
               DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1>
        storage;
    std::array<DeltafinProviderDSparkTensorV1,
               DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1>
        descriptors = {};
    for (std::uint32_t slot = 1;
         slot <= DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1; ++slot) {
      std::uint32_t rank = 0;
      const auto shape = shape_for(slot, rank);
      storage[slot - 1].resize(
          static_cast<std::size_t>(elements(shape, rank)));
      for (std::size_t index = 0; index < storage[slot - 1].size(); ++index) {
        // BF16 encodings for small finite values, including nonzero norms.
        storage[slot - 1][index] = static_cast<std::uint16_t>(
            0x3b80U + ((index + slot) % 16));
      }
      descriptors[slot - 1].slot = slot;
      descriptors[slot - 1].scalar_type =
          DELTAFIN_PROVIDER_DSPARK_BF16_V1;
      descriptors[slot - 1].rank = rank;
      descriptors[slot - 1].shape[0] = shape[0];
      descriptors[slot - 1].shape[1] = shape[1];
      descriptors[slot - 1].data = reinterpret_cast<const std::uint8_t*>(
          storage[slot - 1].data());
      descriptors[slot - 1].data_length =
          storage[slot - 1].size() * sizeof(std::uint16_t);
    }
    std::vector<float> head(32 * 8);
    for (std::size_t index = 0; index < head.size(); ++index) {
      head[index] = static_cast<float>(static_cast<int>(index % 17) - 8) /
                    64.0F;
    }
    DeltafinProviderDSparkCreateV1 create = {};
    create.struct_size = sizeof(create);
    create.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    create.session = session_report.session;
    create.flags = DELTAFIN_PROVIDER_DSPARK_SYNTHETIC_V1;
    create.tensor_count = descriptors.size();
    create.tensors = descriptors.data();
    create.synthetic_head_f32 = head.data();
    create.synthetic_head_elements = head.size();

    auto duplicate = descriptors;
    duplicate[1].slot = 1;
    create.tensors = duplicate.data();
    DeltafinProviderDSparkReportV1 rejected = {};
    rejected.struct_size = sizeof(rejected);
    if (deltafin_provider_dspark_create_v1(
            &create, &rejected, error, sizeof(error)) == 0 ||
        rejected.model != 0) {
      throw std::runtime_error("duplicate DSpark roster did not fail closed");
    }

    auto aliased = descriptors;
    aliased[1].data = aliased[0].data;
    create.tensors = aliased.data();
    rejected = {};
    rejected.struct_size = sizeof(rejected);
    if (deltafin_provider_dspark_create_v1(
            &create, &rejected, error, sizeof(error)) == 0 ||
        rejected.model != 0) {
      throw std::runtime_error("aliased DSpark roster did not fail closed");
    }

    auto wrong_shape = descriptors;
    ++wrong_shape[0].shape[1];
    create.tensors = wrong_shape.data();
    rejected = {};
    rejected.struct_size = sizeof(rejected);
    if (deltafin_provider_dspark_create_v1(
            &create, &rejected, error, sizeof(error)) == 0 ||
        rejected.model != 0) {
      throw std::runtime_error("wrong-shaped DSpark tensor did not fail closed");
    }

    auto wrong_dtype = descriptors;
    wrong_dtype[0].scalar_type = 99;
    create.tensors = wrong_dtype.data();
    rejected = {};
    rejected.struct_size = sizeof(rejected);
    if (deltafin_provider_dspark_create_v1(
            &create, &rejected, error, sizeof(error)) == 0 ||
        rejected.model != 0) {
      throw std::runtime_error("wrong-typed DSpark tensor did not fail closed");
    }

    std::vector<DeltafinProviderDSparkTensorV1> extra(
        descriptors.begin(), descriptors.end());
    extra.push_back(descriptors.front());
    create.tensors = extra.data();
    create.tensor_count = extra.size();
    rejected = {};
    rejected.struct_size = sizeof(rejected);
    if (deltafin_provider_dspark_create_v1(
            &create, &rejected, error, sizeof(error)) == 0 ||
        rejected.model != 0) {
      throw std::runtime_error("extra DSpark tensor did not fail closed");
    }

    create.tensors = descriptors.data();
    create.tensor_count = descriptors.size() - 1;
    rejected = {};
    rejected.struct_size = sizeof(rejected);
    if (deltafin_provider_dspark_create_v1(
            &create, &rejected, error, sizeof(error)) == 0 ||
        rejected.model != 0) {
      throw std::runtime_error("missing DSpark roster did not fail closed");
    }
    create.tensor_count = descriptors.size();
    DeltafinProviderDSparkReportV1 model = {};
    model.struct_size = sizeof(model);
    require_ok(deltafin_provider_dspark_create_v1(
                   &create, &model, error, sizeof(error)),
               error, "create DSpark");
    if (model.model == 0 || model.tensor_count != 67 ||
        model.cache_length != 0 || model.cache_generation != 0) {
      throw std::runtime_error("DSpark create report is invalid");
    }

    std::vector<std::uint16_t> context(3 * 40, 0x3d80U);
    DeltafinProviderTensorUploadBf16V1 upload = {};
    upload.struct_size = sizeof(upload);
    upload.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    upload.session = session_report.session;
    upload.rows = 3;
    upload.columns = 40;
    upload.data = reinterpret_cast<const std::uint8_t*>(context.data());
    upload.byte_length = context.size() * sizeof(std::uint16_t);
    DeltafinProviderTensorReportV1 context_tensor = {};
    context_tensor.struct_size = sizeof(context_tensor);
    require_ok(deltafin_provider_tensor_upload_bf16_v1(
                   &upload, &context_tensor, error, sizeof(error)),
               error, "upload provider-owned DSpark target context");

    DeltafinProviderDSparkAppendTensorV1 append = {};
    append.struct_size = sizeof(append);
    append.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    append.session = session_report.session;
    append.model = model.model;
    append.target_context = context_tensor.tensor;
    append.expected_cache_length = 0;
    append.expected_cache_generation = 0;
    append.rows = 2;
    model.struct_size = sizeof(model);
    require_ok(deltafin_provider_dspark_append_target_tensor_v1(
                   &append, &model, error, sizeof(error)),
               error, "append provider-owned target context");
    if (model.cache_length != 2 || model.cache_generation != 1) {
      throw std::runtime_error("DSpark tensor-append report is invalid");
    }
    DeltafinProviderDSparkReportV1 stale = {};
    stale.struct_size = sizeof(stale);
    if (deltafin_provider_dspark_append_target_tensor_v1(
            &append, &stale, error, sizeof(error)) == 0 ||
        stale.model != 0) {
      throw std::runtime_error(
          "stale DSpark tensor append boundary did not fail closed");
    }

    auto model_resource = resource(session_report.session, model.model);
    DeltafinProviderDSparkSnapshotReportV1 snapshot = {};
    snapshot.struct_size = sizeof(snapshot);
    require_ok(deltafin_provider_dspark_snapshot_v1(
                   &model_resource, &snapshot, error, sizeof(error)),
               error, "snapshot DSpark");

    std::vector<std::uint16_t> embeddings(7 * 8, 0x3e80U);
    DeltafinProviderDSparkProposeV1 propose = {};
    propose.struct_size = sizeof(propose);
    propose.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    propose.session = session_report.session;
    propose.model = model.model;
    propose.anchor_token_id = 4;
    propose.score_rows = 2;
    propose.query_embeddings_bf16 =
        reinterpret_cast<const std::uint8_t*>(embeddings.data());
    propose.query_embedding_bytes = embeddings.size() * sizeof(std::uint16_t);
    DeltafinProviderDSparkProposalReportV1 proposal = {};
    proposal.struct_size = sizeof(proposal);
    require_ok(deltafin_provider_dspark_propose_v1(
                   &propose, &proposal, error, sizeof(error)),
               error, "propose DSpark");
    if (proposal.score_rows != 2 || proposal.anchor_position != 2 ||
        proposal.cache_generation != 1 ||
        !std::isfinite(proposal.confidence_logits[0]) ||
        !std::isfinite(proposal.confidence_logits[1])) {
      throw std::runtime_error("DSpark proposal report is invalid");
    }

    DeltafinProviderDSparkRestoreV1 restore = {};
    restore.struct_size = sizeof(restore);
    restore.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    restore.session = session_report.session;
    restore.model = model.model;
    restore.snapshot = snapshot.snapshot;
    model.struct_size = sizeof(model);
    require_ok(deltafin_provider_dspark_restore_v1(
                   &restore, &model, error, sizeof(error)),
               error, "restore DSpark");
    if (model.cache_length != 2 || model.cache_generation != 2) {
      throw std::runtime_error("DSpark restore report is invalid");
    }

    auto snapshot_resource =
        resource(session_report.session, snapshot.snapshot);
    require_ok(deltafin_provider_dspark_snapshot_destroy_v1(
                   &snapshot_resource, error, sizeof(error)),
               error, "destroy DSpark snapshot");
    auto context_resource =
        resource(session_report.session, context_tensor.tensor);
    require_ok(deltafin_provider_tensor_release_v1(
                   &context_resource, error, sizeof(error)),
               error, "release DSpark target-context tensor");
    require_ok(deltafin_provider_dspark_destroy_v1(
                   &model_resource, error, sizeof(error)),
               error, "destroy DSpark model");
    const auto session_resource = resource(session_report.session, 0);
    require_ok(deltafin_provider_session_destroy_v1(
                   &session_resource, error, sizeof(error)),
               error, "destroy session");
    std::cout << "provider_dspark_abi.synthetic=PASS\n";
    std::cout << "provider_dspark_abi.authority=PROPOSAL_ONLY\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_dspark_abi.synthetic=FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
