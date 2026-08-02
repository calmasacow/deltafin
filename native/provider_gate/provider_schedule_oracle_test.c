/*
 * Independent schedule oracle for the established public K3 path.
 *
 * This test deliberately does not include, link, or call the native provider.
 * Calling provider helpers here would make the implementation its own oracle.
 * Instead it records the observable compiled-operation order, batching shape,
 * provider family, and host synchronization boundaries expressed by:
 *
 *   tools/k3pkg/modeling_kimi_linear.py
 *   tools/fla/modules/__init__.py
 *   tools/fla/ops/kda/__init__.py
 *   tools/kimi_run.py
 *   tools/fast_moe_batch.py
 *   tools/metal_moe.py
 *   tools/cuda_moe.py
 *
 * The production provider can emit the same six-column TSV and use --check to
 * prove that a migration changed only orchestration. The default test covers
 * every public draft/verify width from 1 through 9 and rejects representative
 * regressions: row-wise tensor work, reordered KDA/MLA projections, an early
 * MoE host drain, a fused gate/up GEMM, and position-batched Metal experts.
 */

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_EVENTS 192
#define FIELD_CAP 128
#define LINE_CAP 768

typedef enum Backend {
  BACKEND_CPU,
  BACKEND_MPS,
  BACKEND_CUDA,
} Backend;

typedef enum Stage {
  STAGE_RESIDUAL,
  STAGE_KDA,
  STAGE_MLA,
  STAGE_DENSE,
  STAGE_MOE,
  STAGE_TAIL_PREFILL,
  STAGE_TAIL_VERIFY,
} Stage;

typedef struct Event {
  char scope[FIELD_CAP];
  char operation[FIELD_CAP];
  char shape[FIELD_CAP];
  char provider[FIELD_CAP];
  char boundary[FIELD_CAP];
} Event;

typedef struct Schedule {
  Event events[MAX_EVENTS];
  size_t count;
} Schedule;

static void die(const char *message) {
  fprintf(stderr, "provider schedule oracle: %s\n", message);
  exit(1);
}

static const char *backend_name(const Backend backend) {
  switch (backend) {
    case BACKEND_CPU:
      return "cpu";
    case BACKEND_MPS:
      return "mps";
    case BACKEND_CUDA:
      return "cuda";
  }
  return "invalid";
}

static const char *tensor_provider(const Backend backend) {
  switch (backend) {
    case BACKEND_CPU:
      return "compiled-ATen/CPU";
    case BACKEND_MPS:
      return "compiled-ATen/MPS";
    case BACKEND_CUDA:
      return "compiled-ATen/CUDA";
  }
  return "invalid";
}

static void copy_field(char destination[FIELD_CAP], const char *source) {
  const int written = snprintf(destination, FIELD_CAP, "%s", source);
  if (written < 0 || written >= FIELD_CAP) {
    die("event field exceeds FIELD_CAP");
  }
}

static void add_event(Schedule *schedule, const char *scope,
                      const char *operation, const char *provider,
                      const char *boundary, const char *shape_format, ...) {
  if (schedule->count >= MAX_EVENTS) {
    die("schedule exceeds MAX_EVENTS");
  }
  Event *event = &schedule->events[schedule->count++];
  copy_field(event->scope, scope);
  copy_field(event->operation, operation);
  copy_field(event->provider, provider);
  copy_field(event->boundary, boundary);
  va_list arguments;
  va_start(arguments, shape_format);
  const int written =
      vsnprintf(event->shape, FIELD_CAP, shape_format, arguments);
  va_end(arguments);
  if (written < 0 || written >= FIELD_CAP) {
    die("event shape exceeds FIELD_CAP");
  }
}

static void build_residual(Schedule *schedule, const Backend backend,
                           const size_t width) {
  const char *provider = tensor_provider(backend);
  /* Normal non-boundary decoder layer with A existing residual anchors. */
  add_event(schedule, "residual", "pre_attention_mix", provider, "async",
            "[%zu,A+1,7168]->[%zu,7168]", width, width);
  add_event(schedule, "residual", "input_rms_norm", provider, "async",
            "[%zu,7168]", width);
  add_event(schedule, "residual", "attention_prefix_add", provider, "async",
            "[%zu,7168]+[%zu,7168]", width, width);
  add_event(schedule, "residual", "pre_mlp_mix", provider, "async",
            "[%zu,A+1,7168]->[%zu,7168]", width, width);
  add_event(schedule, "residual", "post_attention_rms_norm", provider,
            "async", "[%zu,7168]", width);
  add_event(schedule, "residual", "mlp_prefix_add", provider, "async",
            "[%zu,7168]+[%zu,7168]", width, width);
}

static void build_kda(Schedule *schedule, const Backend backend,
                      const size_t width) {
  const char *provider = tensor_provider(backend);
  add_event(schedule, "kda", "query_projection", provider, "async",
            "[%zu,7168]x[7168,12288]->[%zu,12288]", width, width);
  add_event(schedule, "kda", "key_projection", provider, "async",
            "[%zu,7168]x[7168,12288]->[%zu,12288]", width, width);
  add_event(schedule, "kda", "value_projection", provider, "async",
            "[%zu,7168]x[7168,12288]->[%zu,12288]", width, width);

  const bool mulsum = backend == BACKEND_CPU || width == 1;
  const char *shortconv_provider = mulsum ? "compiled-ATen/mulsum"
                                           : "compiled-ATen/conv1d";
  add_event(schedule, "kda", "query_shortconv", shortconv_provider, "async",
            "[1,12288,%zu+3]->[1,12288,%zu]", width, width);
  add_event(schedule, "kda", "key_shortconv", shortconv_provider, "async",
            "[1,12288,%zu+3]->[1,12288,%zu]", width, width);
  add_event(schedule, "kda", "value_shortconv", shortconv_provider, "async",
            "[1,12288,%zu+3]->[1,12288,%zu]", width, width);

  add_event(schedule, "kda", "feature_a_projection", provider, "async",
            "[%zu,7168]x[7168,128]->[%zu,128]", width, width);
  add_event(schedule, "kda", "feature_b_projection", provider, "async",
            "[%zu,128]x[128,12288]->[%zu,12288]", width, width);
  add_event(schedule, "kda", "beta_projection", provider, "async",
            "[%zu,7168]x[7168,96]->[%zu,96]", width, width);
  add_event(schedule, "kda", "normalize_query", provider, "async",
            "[1,%zu,96,128]", width);
  add_event(schedule, "kda", "normalize_key", provider, "async",
            "[1,%zu,96,128]", width);
  add_event(schedule, "kda", "beta_sigmoid", provider, "async",
            "[1,%zu,96]", width);
  add_event(schedule, "kda", "gate_transform", provider, "async",
            "[1,%zu,96,128]", width);
  add_event(schedule, "kda", "state_zero_then_initial_add", provider,
            "async", "[1,96,128,128]");
  add_event(schedule, "kda", "query_scale", provider, "async",
            "[1,%zu,96,128]", width);
  for (size_t row = 0; row < width; ++row) {
    char operation[FIELD_CAP];
    snprintf(operation, sizeof(operation), "state_decay[%zu]", row);
    add_event(schedule, "kda", operation, provider, "async",
              "[1,96,128,128]");
    snprintf(operation, sizeof(operation), "delta_reduce[%zu]", row);
    add_event(schedule, "kda", operation, provider, "async",
              "[1,96,128]x[1,96,128,128]->[1,96,128]");
    snprintf(operation, sizeof(operation), "state_outer_add[%zu]", row);
    add_event(schedule, "kda", operation, provider, "async",
              "[1,96,128]outer[1,96,128]->[1,96,128,128]");
    snprintf(operation, sizeof(operation), "output_contract[%zu]", row);
    add_event(schedule, "kda", operation, provider, "async",
              "[1,96,128]x[1,96,128,128]->[1,96,128]");
  }
  add_event(schedule, "kda", "output_gate_projection", provider, "async",
            "[%zu,7168]x[7168,12288]->[%zu,12288]", width, width);
  add_event(schedule, "kda", "output_rms_gate", provider, "async",
            "[1,%zu,96,128]", width);
  add_event(schedule, "kda", "output_projection", provider, "async",
            "[%zu,12288]x[12288,7168]->[%zu,7168]", width, width);
  add_event(schedule, "kda", "cache_publish_after_layer", "host-orchestration",
            "transactional", "one causal boundary per row; final version += %zu",
            width);
}

static void build_mla(Schedule *schedule, const Backend backend,
                      const size_t width) {
  const char *provider = tensor_provider(backend);
  add_event(schedule, "mla", "query_a_projection", provider, "async",
            "[%zu,7168]x[7168,1536]->[%zu,1536]", width, width);
  add_event(schedule, "mla", "query_a_rms_norm", provider, "async",
            "[%zu,1536]", width);
  add_event(schedule, "mla", "query_b_projection", provider, "async",
            "[%zu,1536]x[1536,18432]->[%zu,18432]", width, width);
  add_event(schedule, "mla", "query_view_split", "view-metadata", "none",
            "[1,%zu,96,192]->q_nope[1,96,%zu,128]+q_rope[1,96,%zu,64]",
            width, width, width);
  add_event(schedule, "mla", "key_value_a_projection", provider, "async",
            "[%zu,7168]x[7168,576]->[%zu,576]", width, width);
  add_event(schedule, "mla", "key_value_a_split", "view-metadata", "none",
            "[%zu,576]->kv_lora[%zu,512]+k_rope[%zu,64]", width, width,
            width);
  add_event(schedule, "mla", "key_value_a_rms_norm", provider, "async",
            "[%zu,512]", width);
  add_event(schedule, "mla", "key_value_b_projection", provider, "async",
            "[%zu,512]x[512,24576]->[%zu,24576]", width, width);
  add_event(schedule, "mla", "key_value_view_split", "view-metadata", "none",
            "[1,%zu,96,256]->k_nope[1,96,%zu,128]+v[1,96,%zu,128]",
            width, width, width);
  add_event(schedule, "mla", "rope_expand_and_qk_cat", provider, "async",
            "q,k=[1,96,%zu,192]", width);
  add_event(schedule, "mla", "cache_append", provider, "transactional",
            "keys[1,96,past+%zu,192],values[1,96,past+%zu,128]", width,
            width);
  add_event(schedule, "mla", "query_key_matmul_and_scale", provider, "async",
            "[1,96,%zu,192]x[1,96,192,past+%zu]->[1,96,%zu,past+%zu]",
            width, width, width, width);
  add_event(schedule, "mla", "causal_mask_add", provider, "async",
            "[1,1,%zu,past+%zu], blocked=finfo(fp32).min", width, width);
  add_event(schedule, "mla", "attention_softmax_fp32", provider, "async",
            "[1,96,%zu,past+%zu]", width, width);
  add_event(schedule, "mla", "probability_value_matmul", provider, "async",
            "[1,96,%zu,past+%zu]x[1,96,past+%zu,128]->[1,96,%zu,128]",
            width, width, width, width);
  add_event(schedule, "mla", "output_gate_projection", provider, "async",
            "[%zu,7168]x[7168,12288]->[%zu,12288]", width, width);
  add_event(schedule, "mla", "output_gate_sigmoid_multiply", provider, "async",
            "[%zu,12288]", width);
  add_event(schedule, "mla", "output_projection", provider, "async",
            "[%zu,12288]x[12288,7168]->[%zu,7168]", width, width);
}

static void build_dense(Schedule *schedule, const Backend backend,
                        const size_t width) {
  const char *provider = tensor_provider(backend);
  add_event(schedule, "dense", "gate_projection", provider, "async",
            "[%zu,7168]x[7168,33792]->[%zu,33792]", width, width);
  add_event(schedule, "dense", "up_projection", provider, "async",
            "[%zu,7168]x[7168,33792]->[%zu,33792]", width, width);
  add_event(schedule, "dense", "gate_up_cat", provider, "async",
            "two [%zu,33792]->[%zu,67584]", width, width);
  add_event(schedule, "dense", "situ_activation", provider, "async",
            "[%zu,67584]->[%zu,33792]", width, width);
  add_event(schedule, "dense", "down_projection", provider, "async",
            "[%zu,33792]x[33792,7168]->[%zu,7168]", width, width);
}

static void build_moe(Schedule *schedule, const Backend backend,
                      const size_t width) {
  const char *provider = tensor_provider(backend);
  add_event(schedule, "moe", "router_projection", provider, "async",
            "[%zu,7168]x[7168,896]->[%zu,896]", width, width);
  add_event(schedule, "moe", "router_sigmoid", provider, "async",
            "[%zu,896]", width);
  add_event(schedule, "moe", "router_correction_add", provider, "async",
            "[%zu,896]", width);
  add_event(schedule, "moe", "router_topk_unsorted", provider, "async",
            "[%zu,896]->ids[%zu,16]", width, width);
  add_event(schedule, "moe", "router_gather_uncorrected", provider, "async",
            "scores[%zu,896],ids[%zu,16]->weights[%zu,16]", width, width,
            width);
  add_event(schedule, "moe", "router_weight_sum_plus_1e-20", provider,
            "async", "[%zu,16]->[%zu,1]", width, width);
  add_event(schedule, "moe", "router_weight_divide", provider, "async",
            "[%zu,16]", width);
  add_event(schedule, "moe", "routed_down_projection", provider, "async",
            "[%zu,7168]x[7168,3584]->[%zu,3584]", width, width);
  add_event(schedule, "moe", "route_materialize_ids_and_weights",
            "host-orchestration",
            backend == BACKEND_CPU ? "host-read-no-device-drain"
                                   : "device-to-host-queue-drain",
            "ids[%zu,16]+weights[%zu,16]", width, width);
  add_event(schedule, "moe", "demand_expert_fetch", "pread/I/O", "blocking",
            "ordered union of %zu top-16 routes", width);
  if (backend == BACKEND_MPS) {
    add_event(schedule, "moe", "routed_input_materialize", "ATen MPS->CPU",
              "after-demand-fetch", "[%zu,3584]", width);
    for (size_t row = 0; row < width; ++row) {
      char operation[FIELD_CAP];
      snprintf(operation, sizeof(operation), "expert_metal_row[%zu]", row);
      add_event(schedule, "moe", operation,
                "compiled-Metal/one-command-buffer-per-row", "synchronous-ABI",
                "[1,3584],16 experts,ordered reduction");
    }
    add_event(schedule, "moe", "routed_output_return", "ATen CPU->MPS",
              "after-experts", "[%zu,3584]", width);
  } else if (backend == BACKEND_CPU) {
    for (size_t row = 0; row < width; ++row) {
      char operation[FIELD_CAP];
      snprintf(operation, sizeof(operation), "expert_cpu_row[%zu]", row);
      add_event(schedule, "moe", operation,
                "compiled-C/two-phase+NumPy-ordered-reduce", "synchronous-ABI",
                "[1,3584],16 experts");
    }
  } else {
    add_event(schedule, "moe", "expert_cuda_by_unique_expert",
              "compiled-CUDA/current-stream", "async",
              "U launches over %zu rows; U in [16,min(896,16T)]", width);
    add_event(schedule, "moe", "expert_cuda_ordered_route_reduce",
              "compiled-ATen/CUDA", "async",
              "16 addcmul launches over [%zu,3584]", width);
  }
  add_event(schedule, "moe", "routed_rms_norm", provider, "async",
            "[%zu,3584]", width);
  add_event(schedule, "moe", "routed_up_projection", provider, "async",
            "[%zu,3584]x[3584,7168]->[%zu,7168]", width, width);
  add_event(schedule, "moe", "shared_gate_projection", provider, "async",
            "[%zu,7168]x[7168,6144]->[%zu,6144]", width, width);
  add_event(schedule, "moe", "shared_up_projection", provider, "async",
            "[%zu,7168]x[7168,6144]->[%zu,6144]", width, width);
  add_event(schedule, "moe", "shared_gate_up_cat", provider, "async",
            "two [%zu,6144]->[%zu,12288]", width, width);
  add_event(schedule, "moe", "shared_situ_activation", provider, "async",
            "[%zu,12288]->[%zu,6144]", width, width);
  add_event(schedule, "moe", "shared_down_projection", provider, "async",
            "[%zu,6144]x[6144,7168]->[%zu,7168]", width, width);
  add_event(schedule, "moe", "routed_plus_shared", provider, "async",
            "[%zu,7168]+[%zu,7168]", width, width);
}

static void build_tail(Schedule *schedule, const Backend backend,
                       const size_t width, const bool prefill) {
  const char *provider = tensor_provider(backend);
  add_event(schedule, prefill ? "tail-prefill" : "tail-verify",
            "output_residual_mix", provider, "async",
            "[%zu,A+1,7168]->[%zu,7168]", width, width);
  add_event(schedule, prefill ? "tail-prefill" : "tail-verify",
            "final_rms_norm", provider, "async", "[%zu,7168]", width);
  const size_t head_rows = prefill ? 1 : width;
  add_event(schedule, prefill ? "tail-prefill" : "tail-verify",
            "language_model_head", provider, "async",
            "[%zu,7168]x[7168,163840]->[%zu,163840]", head_rows,
            head_rows);
}

static Schedule build_schedule(const Backend backend, const size_t width,
                               const Stage stage) {
  Schedule schedule = {0};
  switch (stage) {
    case STAGE_RESIDUAL:
      build_residual(&schedule, backend, width);
      break;
    case STAGE_KDA:
      build_kda(&schedule, backend, width);
      break;
    case STAGE_MLA:
      build_mla(&schedule, backend, width);
      break;
    case STAGE_DENSE:
      build_dense(&schedule, backend, width);
      break;
    case STAGE_MOE:
      build_moe(&schedule, backend, width);
      break;
    case STAGE_TAIL_PREFILL:
      build_tail(&schedule, backend, width, true);
      break;
    case STAGE_TAIL_VERIFY:
      build_tail(&schedule, backend, width, false);
      break;
  }
  return schedule;
}

static ptrdiff_t find_operation(const Schedule *schedule,
                                const char *operation) {
  for (size_t index = 0; index < schedule->count; ++index) {
    if (strcmp(schedule->events[index].operation, operation) == 0) {
      return (ptrdiff_t)index;
    }
  }
  return -1;
}

static void require_before(const Schedule *schedule, const char *first,
                           const char *second) {
  const ptrdiff_t first_index = find_operation(schedule, first);
  const ptrdiff_t second_index = find_operation(schedule, second);
  if (first_index < 0 || second_index < 0 || first_index >= second_index) {
    fprintf(stderr, "expected %s before %s\n", first, second);
    exit(1);
  }
}

static bool event_equal(const Event *left, const Event *right) {
  return strcmp(left->scope, right->scope) == 0 &&
         strcmp(left->operation, right->operation) == 0 &&
         strcmp(left->shape, right->shape) == 0 &&
         strcmp(left->provider, right->provider) == 0 &&
         strcmp(left->boundary, right->boundary) == 0;
}

static bool schedule_equal(const Schedule *expected, const Schedule *actual,
                           const bool diagnose) {
  if (expected->count != actual->count) {
    if (diagnose) {
      fprintf(stderr, "event count: expected %zu, observed %zu\n",
              expected->count, actual->count);
    }
    return false;
  }
  for (size_t index = 0; index < expected->count; ++index) {
    if (!event_equal(&expected->events[index], &actual->events[index])) {
      if (diagnose) {
        const Event *want = &expected->events[index];
        const Event *got = &actual->events[index];
        fprintf(stderr,
                "event %zu mismatch\n  expected: %s | %s | %s | %s | %s\n"
                "  observed: %s | %s | %s | %s | %s\n",
                index, want->scope, want->operation, want->shape,
                want->provider, want->boundary, got->scope, got->operation,
                got->shape, got->provider, got->boundary);
      }
      return false;
    }
  }
  return true;
}

static void swap_events(Schedule *schedule, const char *left,
                        const char *right) {
  const ptrdiff_t left_index = find_operation(schedule, left);
  const ptrdiff_t right_index = find_operation(schedule, right);
  if (left_index < 0 || right_index < 0) {
    die("mutation requested an absent event");
  }
  const Event temporary = schedule->events[left_index];
  schedule->events[left_index] = schedule->events[right_index];
  schedule->events[right_index] = temporary;
}

static void require_rejected_mutation(const Schedule *expected,
                                      Schedule mutated, const char *name) {
  if (schedule_equal(expected, &mutated, false)) {
    fprintf(stderr, "oracle accepted forbidden mutation: %s\n", name);
    exit(1);
  }
}

static void validate_one(const Backend backend, const size_t width) {
  const Schedule residual = build_schedule(backend, width, STAGE_RESIDUAL);
  for (size_t index = 0; index < residual.count; ++index) {
    char needle[32];
    snprintf(needle, sizeof(needle), "[%zu", width);
    if (strstr(residual.events[index].shape, needle) == NULL) {
      die("residual event is not full-T");
    }
  }

  const Schedule kda = build_schedule(backend, width, STAGE_KDA);
  require_before(&kda, "value_projection", "query_shortconv");
  require_before(&kda, "value_shortconv", "feature_a_projection");
  require_before(&kda, "feature_b_projection", "beta_projection");
  require_before(&kda, "output_contract[0]", "output_gate_projection");
  require_before(&kda, "output_gate_projection", "output_projection");

  const Schedule mla = build_schedule(backend, width, STAGE_MLA);
  require_before(&mla, "query_a_projection", "query_a_rms_norm");
  require_before(&mla, "query_a_rms_norm", "query_b_projection");
  require_before(&mla, "query_b_projection", "key_value_a_projection");
  require_before(&mla, "attention_softmax_fp32",
                 "output_gate_projection");
  require_before(&mla, "output_gate_projection", "output_projection");

  const Schedule dense = build_schedule(backend, width, STAGE_DENSE);
  require_before(&dense, "gate_projection", "up_projection");
  require_before(&dense, "up_projection", "gate_up_cat");

  const Schedule moe = build_schedule(backend, width, STAGE_MOE);
  require_before(&moe, "routed_down_projection",
                 "route_materialize_ids_and_weights");
  require_before(&moe, "route_materialize_ids_and_weights",
                 "demand_expert_fetch");
  require_before(&moe, "routed_up_projection", "shared_gate_projection");
  require_before(&moe, "shared_gate_projection", "shared_up_projection");
  require_before(&moe, "shared_up_projection", "shared_gate_up_cat");

  if (backend == BACKEND_MPS) {
    require_before(&moe, "demand_expert_fetch", "routed_input_materialize");
    size_t metal_rows = 0;
    for (size_t index = 0; index < moe.count; ++index) {
      if (strncmp(moe.events[index].operation, "expert_metal_row[", 17) == 0) {
        ++metal_rows;
      }
    }
    if (metal_rows != width) {
      die("default Metal schedule is not one expert command per row");
    }
  }

  const Schedule tail_prefill =
      build_schedule(backend, width, STAGE_TAIL_PREFILL);
  const Schedule tail_verify =
      build_schedule(backend, width, STAGE_TAIL_VERIFY);
  if (strstr(tail_prefill.events[2].shape, "[1,7168]") == NULL) {
    die("prefill tail did not select one last row only at the head");
  }
  char verify_head[32];
  snprintf(verify_head, sizeof(verify_head), "[%zu,7168]", width);
  if (strstr(tail_verify.events[2].shape, verify_head) == NULL) {
    die("verify tail did not retain every row at the head");
  }

  /* Falsifiers for the exact classes of migration mistake found in audit. */
  Schedule mutation = kda;
  swap_events(&mutation, "query_shortconv", "feature_a_projection");
  require_rejected_mutation(&kda, mutation,
                            "KDA controller projection before shortconv");
  mutation = mla;
  swap_events(&mutation, "query_a_rms_norm", "output_gate_projection");
  require_rejected_mutation(&mla, mutation, "MLA early output gate");
  mutation = moe;
  swap_events(&mutation, "routed_down_projection",
              "route_materialize_ids_and_weights");
  require_rejected_mutation(&moe, mutation, "MoE early route queue drain");
  mutation = dense;
  copy_field(mutation.events[0].operation, "fused_gate_up_projection");
  require_rejected_mutation(&dense, mutation, "unqualified dense fusion");
  mutation = residual;
  snprintf(mutation.events[0].shape, FIELD_CAP,
           "[1,A+1,7168]->[1,7168] repeated %zu", width);
  require_rejected_mutation(&residual, mutation, "row-wise residual provider");
  if (backend == BACKEND_MPS && width > 1) {
    mutation = moe;
    const ptrdiff_t first = find_operation(&mutation, "expert_metal_row[0]");
    if (first < 0) {
      die("Metal falsifier could not find row zero");
    }
    copy_field(mutation.events[first].operation,
               "expert_metal_positions_fused");
    require_rejected_mutation(&moe, mutation,
                              "unqualified Metal position batch");
  }
  if (backend == BACKEND_MPS) {
    mutation = moe;
    swap_events(&mutation, "demand_expert_fetch",
                "routed_input_materialize");
    require_rejected_mutation(
        &moe, mutation,
        "MPS routed-input transfer before demand expert fetch");
  }
}

static void dump_schedule(const Schedule *schedule) {
  puts("index\tscope\toperation\tshape\tprovider\tboundary");
  for (size_t index = 0; index < schedule->count; ++index) {
    const Event *event = &schedule->events[index];
    printf("%zu\t%s\t%s\t%s\t%s\t%s\n", index, event->scope,
           event->operation, event->shape, event->provider, event->boundary);
  }
}

static bool parse_backend(const char *text, Backend *backend) {
  if (strcmp(text, "cpu") == 0) {
    *backend = BACKEND_CPU;
    return true;
  }
  if (strcmp(text, "mps") == 0) {
    *backend = BACKEND_MPS;
    return true;
  }
  if (strcmp(text, "cuda") == 0) {
    *backend = BACKEND_CUDA;
    return true;
  }
  return false;
}

static bool parse_stage(const char *text, Stage *stage) {
  static const struct {
    const char *name;
    Stage stage;
  } values[] = {{"residual", STAGE_RESIDUAL},
                {"kda", STAGE_KDA},
                {"mla", STAGE_MLA},
                {"dense", STAGE_DENSE},
                {"moe", STAGE_MOE},
                {"tail-prefill", STAGE_TAIL_PREFILL},
                {"tail-verify", STAGE_TAIL_VERIFY}};
  for (size_t index = 0; index < sizeof(values) / sizeof(values[0]); ++index) {
    if (strcmp(text, values[index].name) == 0) {
      *stage = values[index].stage;
      return true;
    }
  }
  return false;
}

static bool parse_width(const char *text, size_t *width) {
  errno = 0;
  char *end = NULL;
  const unsigned long value = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' ||
      value < 1 || value > 9) {
    return false;
  }
  *width = (size_t)value;
  return true;
}

static bool parse_trace(const char *path, Schedule *schedule) {
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "open trace %s: %s\n", path, strerror(errno));
    return false;
  }
  char line[LINE_CAP];
  while (fgets(line, sizeof(line), file) != NULL) {
    if (strncmp(line, "index\t", 6) == 0 || line[0] == '#') {
      continue;
    }
    if (schedule->count >= MAX_EVENTS) {
      fclose(file);
      die("observed trace exceeds MAX_EVENTS");
    }
    char *fields[6] = {0};
    char *cursor = line;
    for (size_t index = 0; index < 6; ++index) {
      fields[index] = cursor;
      char *tab = strchr(cursor, index == 5 ? '\n' : '\t');
      if (tab == NULL) {
        fclose(file);
        fprintf(stderr, "invalid TSV trace line: %s\n", line);
        return false;
      }
      *tab = '\0';
      cursor = tab + 1;
    }
    char *end = NULL;
    const unsigned long observed_index = strtoul(fields[0], &end, 10);
    if (end == fields[0] || *end != '\0' ||
        observed_index != schedule->count) {
      fclose(file);
      return false;
    }
    Event *event = &schedule->events[schedule->count++];
    copy_field(event->scope, fields[1]);
    copy_field(event->operation, fields[2]);
    copy_field(event->shape, fields[3]);
    copy_field(event->provider, fields[4]);
    copy_field(event->boundary, fields[5]);
  }
  const bool okay = !ferror(file);
  fclose(file);
  return okay;
}

static void usage(const char *program) {
  fprintf(stderr,
          "usage:\n"
          "  %s\n"
          "  %s --dump cpu|mps|cuda 1..9 "
          "residual|kda|mla|dense|moe|tail-prefill|tail-verify\n"
          "  %s --check cpu|mps|cuda 1..9 STAGE TRACE.tsv\n",
          program, program, program);
}

int main(const int argc, char **argv) {
  if (argc == 1) {
    static const size_t widths[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    static const Backend backends[] = {BACKEND_CPU, BACKEND_MPS, BACKEND_CUDA};
    for (size_t backend = 0;
         backend < sizeof(backends) / sizeof(backends[0]); ++backend) {
      for (size_t width = 0; width < sizeof(widths) / sizeof(widths[0]);
           ++width) {
        validate_one(backends[backend], widths[width]);
      }
    }
    puts("provider_schedule_oracle=PASS backends=cpu,mps,cuda widths=1..9");
    return 0;
  }

  const bool dump = argc == 5 && strcmp(argv[1], "--dump") == 0;
  const bool check = argc == 6 && strcmp(argv[1], "--check") == 0;
  if (!dump && !check) {
    usage(argv[0]);
    return 2;
  }
  Backend backend;
  size_t width = 0;
  Stage stage;
  if (!parse_backend(argv[2], &backend) || !parse_width(argv[3], &width) ||
      !parse_stage(argv[4], &stage)) {
    usage(argv[0]);
    return 2;
  }
  const Schedule expected = build_schedule(backend, width, stage);
  if (dump) {
    dump_schedule(&expected);
    return 0;
  }
  Schedule observed = {0};
  if (!parse_trace(argv[5], &observed)) {
    return 2;
  }
  if (!schedule_equal(&expected, &observed, true)) {
    return 1;
  }
  printf("provider_schedule_oracle=PASS backend=%s width=%zu\n",
         backend_name(backend), width);
  return 0;
}
