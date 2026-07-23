#!/usr/bin/env bash
# Shared repository-relative paths for M20 experiment entry points.

M20_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
M20_ROOT=$(cd -- "${M20_SCRIPT_DIR}/.." && pwd)
M20_PROJECT_ROOT=$(cd -- "${M20_ROOT}/../.." && pwd)

M20_PYTHON=${PYTHON_BIN:-"${M20_PROJECT_ROOT}/.venv_kt/bin/python"}
if [[ ! -x "${M20_PYTHON}" && -x "${M20_PROJECT_ROOT}/../.venv_kt/bin/python" ]]; then
  # Compatibility for the original workspace, where this project was nested.
  M20_PYTHON="${M20_PROJECT_ROOT}/../.venv_kt/bin/python"
fi

M20_MODEL=${MODEL:-"${M20_PROJECT_ROOT}/assets/model_shim_qwen3"}
M20_EXPERIMENTS=${EXPERIMENTS:-"${M20_PROJECT_ROOT}/experiments"}
