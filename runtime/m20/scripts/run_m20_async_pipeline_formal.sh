#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"
ROOT="$M20_PROJECT_ROOT"
PYTHON="$M20_PYTHON"
M20="$M20_ROOT"
MODEL="$M20_MODEL"
EXPERIMENTS="$M20_EXPERIMENTS"
source "$M20/scripts/host_resource_gate.sh"
source "$M20/scripts/gpu_resource_gate.sh"

# Resolve symlinked repository assets before passing them to the runtime.  The
# KT LLAMAFILE loader identifies a GGUF input from its filename suffix, so the
# stable assets/qwen3_gguf convenience link must become the real .gguf path.
ASSET_ROOT="$ROOT/assets"
GGUF=$(readlink -f "${GGUF:-$ASSET_ROOT/qwen3_gguf}")
PROMPT_FILE=$(readlink -f "${PROMPT_FILE:-$ASSET_ROOT/workloads/text/sharegpt_long_qwen3_min2048_512.jsonl}")
PROMPT_IDENTITY_MANIFEST=$(readlink -f "${PROMPT_IDENTITY_MANIFEST:-$ASSET_ROOT/workload_identity/sharegpt_long_qwen3_min2048_512.json}")
ORACLE_TRACE=$(readlink -f "${ORACLE_TRACE:-$ASSET_ROOT/kt_native_oracle_stats/kt_llamafile_tp1_seq2048_128p_gpu64_uniform_sharegpt_long_seq2048_test128_offset128_c4_cps256_top4rec/routed_experts_trace.jsonl}")
ACTIVATION_STATS=$(readlink -f "${ACTIVATION_STATS:-$ASSET_ROOT/kt_native_oracle_stats/sharegpt_long_seq2048_train128_per_layer_top4_activation_stats.pt}")

GPU=${GPU:-2}
CPUSET=${CPUSET:-0-63}
BASE_PORT=${BASE_PORT:-33900}
OUTPUT_DIR=${OUTPUT_DIR:-$EXPERIMENTS/m20_b_async_pipeline_formal_g4s8k1_p8c8_20260722}
GROUP_SIZE=${GROUP_SIZE:-4}
SLOTS=${SLOTS:-8}
MAX_REPLACEMENTS=${MAX_REPLACEMENTS:-1}
COHORT_SIZE=${COHORT_SIZE:-4}
CANDIDATE_WINDOW=${CANDIDATE_WINDOW:-8}
MAX_CONSECUTIVE=${MAX_CONSECUTIVE:-4}
MAX_WAIT_MS=${MAX_WAIT_MS:-0}
MAX_INFLIGHT_CHUNKS=${MAX_INFLIGHT_CHUNKS:-8}
EXCLUSIVE_LOCK_PATH=${EXCLUSIVE_LOCK_PATH:-/tmp/m20_b_async_pipeline_formal.lock}
WAIT_FOR_HOST_QUIET=${WAIT_FOR_HOST_QUIET:-1}
WAIT_FOR_GPU_QUIET=${WAIT_FOR_GPU_QUIET:-1}
HOST_QUIET_REQUIRED_PASSES=${HOST_QUIET_REQUIRED_PASSES:-2}
GPU_QUIET_REQUIRED_PASSES=${GPU_QUIET_REQUIRED_PASSES:-3}

H2D_MS=${H2D_MS:-1.427118586964988}
D2D_MS=${D2D_MS:-0.10408825399043947}
ROUTE_MS=${ROUTE_MS:-0.005176338825467556}

runner_args=(
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL"
  --gguf "$GGUF"
  --prompt-file "$PROMPT_FILE"
  --prompt-identity-manifest "$PROMPT_IDENTITY_MANIFEST"
  --oracle-trace "$ORACLE_TRACE"
  --activation-stats "$ACTIVATION_STATS"
  --group-size "$GROUP_SIZE"
  --slots "$SLOTS"
  --max-replacements "$MAX_REPLACEMENTS"
  --placement oracle
  --policies min_delta
  --cohort-size "$COHORT_SIZE"
  --candidate-window "$CANDIDATE_WINDOW"
  --max-consecutive "$MAX_CONSECUTIVE"
  --max-wait-ms "$MAX_WAIT_MS"
  --max-inflight-chunks "$MAX_INFLIGHT_CHUNKS"
  --stage-h2d-expert-ms "$H2D_MS"
  --stage-d2d-expert-ms "$D2D_MS"
  --stage-route-entry-gain-ms "$ROUTE_MS"
  --require-physical-paths
  --replay-load-mode both
  --resource-mode exclusive
  --exclusive-lock-path "$EXCLUSIVE_LOCK_PATH"
  --seq-len 2048
  --chunked-prefill-size 256
  --num-prompts 8
  --prompt-offset 128
  --warmup-num-prompts 8
  --warmup-prompt-offset 136
  --warmup-concurrency 8
  --concurrency 8
  --max-total-tokens 40000
  --cpu-threads 64
  --threadpool-count 2
  --cpu-tensor-cache-items 256
  --pin-cpu-tensors
  --mem-fraction-static 0.7
  --gpu "$GPU"
  --base-port "$BASE_PORT"
  --server-timeout-s 1200
  --client-timeout-s 1200
  --timing-log-interval 1
)

if find "$MODEL" -maxdepth 1 \( -name '*.part' -o -name '*.aria2' \) -print -quit | grep -q .; then
  echo "model shim still contains partial downloads" >&2
  exit 1
fi
shard_count=$(find -L "$MODEL" -maxdepth 1 -type f -name 'model-*-of-00016.safetensors' | wc -l)
if [[ "$shard_count" -ne 16 ]]; then
  echo "model shim has $shard_count/16 final shards" >&2
  exit 1
fi
for asset in "$GGUF" "$PROMPT_FILE" "$PROMPT_IDENTITY_MANIFEST" "$ORACLE_TRACE" "$ACTIVATION_STATS"; do
  if [[ ! -f "$asset" ]]; then
    echo "required formal asset is missing or not a file: $asset" >&2
    exit 1
  fi
done
if [[ "$GGUF" != *.gguf ]]; then
  echo "formal GGUF path must retain its .gguf suffix: $GGUF" >&2
  exit 1
fi

bash "$M20/scripts/install_runtime.sh" check

# Freeze the complete three-repeat contract before incremental execution. A
# later --resume keeps this provenance even while each invocation only extends
# the completed prefix by one repeat.
if [[ ! -f "$OUTPUT_DIR/provenance.json" ]]; then
  taskset -c "$CPUSET" "$PYTHON" \
    "$M20/inter_layer_predictor/run_m20_b.py" \
    "${runner_args[@]}" --repeats 3 --plan-only >/dev/null
fi

target_repeat=1
while ((target_repeat <= 3)); do
  while true; do
    wait_for_host_quiet
    wait_for_gpu_quiet "$GPU"

    status=0
    taskset -c "$CPUSET" "$PYTHON" \
      "$M20/inter_layer_predictor/run_m20_b.py" \
      "${runner_args[@]}" --repeats "$target_repeat" --resume || status=$?

    acceptance="$OUTPUT_DIR/pipeline_acceptance.json"
    if [[ -f "$acceptance" ]] && jq -e \
      --argjson expected "$target_repeat" \
      '.complete and .correctness_passed and .resource_audit.passed and
       (.expected_pairs == $expected)' "$acceptance" >/dev/null; then
      break
    fi
    if [[ -f "$acceptance" ]] && jq -e \
      '.complete and (.correctness_passed | not)' "$acceptance" >/dev/null; then
      echo "pipeline correctness failed; refusing to convert it into a retry" >&2
      exit 1
    fi
    if [[ -f "$acceptance" ]] && jq -e \
      '.resource_audit.passed | not' "$acceptance" >/dev/null; then
      invalid_dir="${OUTPUT_DIR}_resource_invalid_$(date -u +%H%M%S)"
      mv "$OUTPUT_DIR" "$invalid_dir"
      echo "archived GPU-contaminated suite at $invalid_dir; restarting" >&2
      taskset -c "$CPUSET" "$PYTHON" \
        "$M20/inter_layer_predictor/run_m20_b.py" \
        "${runner_args[@]}" --repeats 3 --plan-only >/dev/null
      target_repeat=1
      continue
    fi
    echo "repeat prefix $target_repeat incomplete (runner status $status); retrying" >&2
  done
  target_repeat=$((target_repeat + 1))
done

# Rebuild the final three-repeat view without launching another server.
status=0
taskset -c "$CPUSET" "$PYTHON" \
  "$M20/inter_layer_predictor/run_m20_b.py" \
  "${runner_args[@]}" --repeats 3 --summarize-existing || status=$?

if ! jq -e '.complete and .correctness_passed and .formal_eligible' \
  "$OUTPUT_DIR/pipeline_acceptance.json" >/dev/null; then
  echo "formal suite is incomplete or resource-ineligible" >&2
  exit 1
fi

jq '{passed, correctness_passed, formal_eligible, performance_passed,
  throughput_speedup_mean_pct, throughput_speedup_worst_pct}' \
  "$OUTPUT_DIR/pipeline_acceptance.json"

# A negative performance result is a completed acceptance result, not an
# orchestration failure. The JSON/report carry the pass/fail decision.
exit 0
