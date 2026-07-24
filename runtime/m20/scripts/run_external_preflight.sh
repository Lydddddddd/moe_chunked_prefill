#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"
OUTPUT_DIR=${OUTPUT_DIR:-"${M20_PROJECT_ROOT}/reproduction/preflight"}
EXTERNAL_RUNTIME_DIR=${EXTERNAL_RUNTIME_DIR:-"${M20_PROJECT_ROOT}/environment/external_runtime"}
GPU=${GPU:-0}
if [[ -d "${EXTERNAL_RUNTIME_DIR}/lib" ]]; then
  export LD_LIBRARY_PATH="${EXTERNAL_RUNTIME_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "${OUTPUT_DIR}"

exec > >(tee "${OUTPUT_DIR}/preflight.log") 2>&1

echo "commit=$(git -C "${M20_PROJECT_ROOT}" rev-parse HEAD)"
if [[ -n $(git -C "${M20_PROJECT_ROOT}" status --porcelain) ]]; then
  echo "working tree must be clean for external reproduction" >&2
  git -C "${M20_PROJECT_ROOT}" status --short >&2
  exit 1
fi

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  "${M20_PYTHON}" --version
  uname -a
  lscpu
  nvidia-smi -L
  nvidia-smi
} > "${OUTPUT_DIR}/system_info.txt"

"${M20_PYTHON}" -m pip freeze | sort > "${OUTPUT_DIR}/packages.txt"
sha256sum \
  "${M20_ROOT}/sglang/srt/layers/moe/kt_ep_wrapper.py" \
  "${M20_ROOT}/sglang/srt/layers/moe/kt_group_expert_buffer.py" \
  "${M20_ROOT}/sglang/srt/model_executor/model_runner.py" \
  "${M20_ROOT}/sglang/srt/model_executor/kt_stage_batch.py" \
  "${M20_ROOT}/sglang/srt/managers/scheduler.py" \
  "${M20_ROOT}/sglang/srt/managers/kt_stage_scheduler.py" \
  "${M20_ROOT}/sglang/srt/server_args.py" \
  > "${OUTPUT_DIR}/runtime_sha256.txt"

KT_ROOT=$("${M20_PYTHON}" - <<'PY'
import importlib.metadata as md
from pathlib import Path

print(Path(md.distribution("kt-kernel").locate_file("kt_kernel")).resolve())
PY
)
sha256sum \
  "${KT_ROOT}/__init__.py" \
  "${KT_ROOT}/experts.py" \
  "${KT_ROOT}/experts_base.py" \
  "${KT_ROOT}/utils/llamafile.py" \
  "${KT_ROOT}/_kt_kernel_ext_avx2.cpython-310-x86_64-linux-gnu.so" \
  > "${OUTPUT_DIR}/kt_kernel_sha256.txt"

PYTHON_BIN="${M20_PYTHON}" EXTERNAL_RUNTIME_DIR="${EXTERNAL_RUNTIME_DIR}" \
  bash "${M20_PROJECT_ROOT}/environment/verify_external_environment.sh"
bash "${M20_ROOT}/scripts/install_runtime.sh" check
GPU="${GPU}" REQUIRE_CUDA=1 bash "${M20_ROOT}/scripts/run_m20_smoke_tests.sh"

echo "external M20 preflight: PASS"
