#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"
GPU=${GPU:-0}
REQUIRE_CUDA=${REQUIRE_CUDA:-0}

tests=(
  "kt_group_buffer_smoke.py"
  "kt_runtime_mapping_smoke.py"
  "m20_b_runner_smoke.py"
  "m20_b_summary_smoke.py"
  "m20_cost_calibration_smoke.py"
  "m20_pipeline_cuda_smoke.py"
  "m20_runner_logic_smoke.py"
  "m20_server_args_smoke.py"
  "m20_stage_batch_smoke.py"
  "m20_stage_runtime_smoke.py"
  "m20_stage_scheduler_smoke.py"
)

[[ -x "${M20_PYTHON}" ]] || { echo "missing Python: ${M20_PYTHON}" >&2; exit 1; }
if [[ "${REQUIRE_CUDA}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${M20_PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA smoke requires a visible CUDA GPU")
print(f"CUDA preflight: PASS ({torch.cuda.get_device_name(0)})")
PY
fi

for test_name in "${tests[@]}"; do
  test_path="${M20_ROOT}/tests/${test_name}"
  [[ -f "${test_path}" ]] || { echo "missing smoke test: ${test_path}" >&2; exit 1; }
  if [[ "${test_name}" == "m20_pipeline_cuda_smoke.py" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${M20_PYTHON}" "${test_path}"
  else
    "${M20_PYTHON}" "${test_path}"
  fi
done

echo "M20-B smoke suite: PASS (${#tests[@]}/${#tests[@]})"
