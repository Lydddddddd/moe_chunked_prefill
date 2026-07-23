#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
M20_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd -- "${M20_ROOT}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${PROJECT_ROOT}/.venv_kt/bin/python"}
if [[ ! -x "${PYTHON_BIN}" && -x "${PROJECT_ROOT}/../.venv_kt/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/../.venv_kt/bin/python"
fi
MODE=${1:-install}

SITE_ROOT=$(
  "${PYTHON_BIN}" -c 'from pathlib import Path; import sglang; print(Path(sglang.__file__).resolve().parent)'
)

relative_files=(
  "sglang/srt/layers/moe/kt_ep_wrapper.py"
  "sglang/srt/layers/moe/kt_group_expert_buffer.py"
  "sglang/srt/model_executor/model_runner.py"
  "sglang/srt/model_executor/kt_stage_batch.py"
  "sglang/srt/managers/scheduler.py"
  "sglang/srt/managers/kt_stage_scheduler.py"
  "sglang/srt/server_args.py"
)

case "${MODE}" in
  install)
    source_root="${M20_ROOT}"
    ;;
  restore)
    source_root="${M20_ROOT}/upstream_snapshot"
    relative_files=(
      "sglang/srt/layers/moe/kt_ep_wrapper.py"
      "sglang/srt/model_executor/model_runner.py"
      "sglang/srt/managers/scheduler.py"
      "sglang/srt/server_args.py"
    )
    ;;
  check)
    source_root="${M20_ROOT}"
    ;;
  *)
    echo "usage: $0 [install|check|restore]" >&2
    exit 2
    ;;
esac

for relative in "${relative_files[@]}"; do
  source_file="${source_root}/${relative}"
  target_file="${SITE_ROOT}/${relative#sglang/}"
  if [[ ! -f "${source_file}" ]]; then
    echo "missing source file: ${source_file}" >&2
    exit 1
  fi
  if [[ "${MODE}" == "check" ]]; then
    cmp --silent "${source_file}" "${target_file}" || {
      echo "runtime mismatch: ${relative}" >&2
      exit 1
    }
  else
    install -D -m 0644 "${source_file}" "${target_file}"
  fi
done

if [[ "${MODE}" != "check" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    "${SITE_ROOT}/srt/layers/moe/kt_ep_wrapper.py" \
    "${SITE_ROOT}/srt/layers/moe/kt_group_expert_buffer.py" \
    "${SITE_ROOT}/srt/model_executor/model_runner.py" \
    "${SITE_ROOT}/srt/model_executor/kt_stage_batch.py" \
    "${SITE_ROOT}/srt/managers/scheduler.py" \
    "${SITE_ROOT}/srt/managers/kt_stage_scheduler.py" \
    "${SITE_ROOT}/srt/server_args.py"
fi

echo "M20 runtime ${MODE}: PASS (${SITE_ROOT})"
