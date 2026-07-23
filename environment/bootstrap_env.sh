#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.10}
VENV_DIR=${VENV_DIR:-"${ROOT}/.venv_kt"}

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT}/environment/requirements.txt"

cat <<'EOF'
Base environment installed.

Before running M20, provision the approved kt-kernel patch and external assets,
then install the runtime patch with:

  bash runtime/m20/scripts/install_runtime.sh install

See environment/README.md and assets/README.md for the required artifact layout.
EOF
