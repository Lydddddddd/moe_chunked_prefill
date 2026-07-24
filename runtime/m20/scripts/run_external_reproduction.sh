#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"
GPU=${GPU:-0}
CPUSET=${CPUSET:-0-63}
CPU_THREADS=${CPU_THREADS:-64}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-"${M20_PROJECT_ROOT}/reproduction/m20_external_${RUN_TAG}"}
SCREEN_PORT=${SCREEN_PORT:-33900}
FORMAL_PORT=${FORMAL_PORT:-34000}
LOCK_PATH=${EXCLUSIVE_LOCK_PATH:-"/tmp/m20_external_reproduction_gpu${GPU}.lock"}

mkdir -p "${RUN_ROOT}"

GPU="${GPU}" OUTPUT_DIR="${RUN_ROOT}/preflight" \
  bash "${M20_ROOT}/scripts/run_external_preflight.sh"

GPU="${GPU}" CPUSET="${CPUSET}" CPU_THREADS="${CPU_THREADS}" \
REPEATS=1 BASE_PORT="${SCREEN_PORT}" EXCLUSIVE_LOCK_PATH="${LOCK_PATH}" \
OUTPUT_DIR="${RUN_ROOT}/screen_g4s8k1q2" \
  bash "${M20_ROOT}/scripts/run_m20_async_pipeline_formal.sh"

GPU="${GPU}" CPUSET="${CPUSET}" CPU_THREADS="${CPU_THREADS}" \
REPEATS=3 BASE_PORT="${FORMAL_PORT}" EXCLUSIVE_LOCK_PATH="${LOCK_PATH}" \
OUTPUT_DIR="${RUN_ROOT}/formal_g4s8k1q2_r3" \
  bash "${M20_ROOT}/scripts/run_m20_async_pipeline_formal.sh"

jq '{complete, correctness_passed, formal_eligible, performance_passed,
  throughput_speedup_mean_pct, throughput_speedup_worst_pct}' \
  "${RUN_ROOT}/formal_g4s8k1q2_r3/pipeline_acceptance.json"
echo "external reproduction complete: ${RUN_ROOT}"
