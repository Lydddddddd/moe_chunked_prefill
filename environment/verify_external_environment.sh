#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${ROOT}/.venv_kt/bin/python"}
if [[ ! -x "${PYTHON_BIN}" && -x "${ROOT}/../.venv_kt/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/../.venv_kt/bin/python"
fi
EXTERNAL_RUNTIME_DIR=${EXTERNAL_RUNTIME_DIR:-"${ROOT}/environment/external_runtime"}
if [[ -d "${EXTERNAL_RUNTIME_DIR}/lib" ]]; then
  hwloc_path="${EXTERNAL_RUNTIME_DIR}/lib/libhwloc.so.15.5.2"
  [[ -f "${hwloc_path}" ]] || { echo "missing hwloc library: ${hwloc_path}" >&2; exit 1; }
  export LD_LIBRARY_PATH="${EXTERNAL_RUNTIME_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

ASSET_ROOT=${MOE_ASSET_ROOT:-"${ROOT}/assets"}
MODEL=${MODEL:-"${ASSET_ROOT}/model_shim_qwen3"}

resolve_file() {
  local label=$1
  local candidate=$2
  local resolved
  resolved=$(readlink -f -- "${candidate}" 2>/dev/null || true)
  if [[ -z "${resolved}" || ! -f "${resolved}" ]]; then
    echo "missing ${label}: ${candidate}" >&2
    exit 1
  fi
  printf '%s\n' "${resolved}"
}

GGUF=$(resolve_file "GGUF" "${GGUF:-${ASSET_ROOT}/qwen3_gguf}")
WORKLOAD=$(resolve_file "workload" "${PROMPT_FILE:-${ASSET_ROOT}/workloads/text/sharegpt_long_qwen3_min2048_512.jsonl}")
IDENTITY=$(resolve_file "prompt identity manifest" "${PROMPT_IDENTITY_MANIFEST:-${ASSET_ROOT}/workload_identity/sharegpt_long_qwen3_min2048_512.json}")
ORACLE=$(resolve_file "oracle trace" "${ORACLE_TRACE:-${ASSET_ROOT}/kt_native_oracle_stats/kt_llamafile_tp1_seq2048_128p_gpu64_uniform_sharegpt_long_seq2048_test128_offset128_c4_cps256_top4rec/routed_experts_trace.jsonl}")
STATS=$(resolve_file "activation stats" "${ACTIVATION_STATS:-${ASSET_ROOT}/kt_native_oracle_stats/sharegpt_long_seq2048_train128_per_layer_top4_activation_stats.pt}")

check_hash() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "hash mismatch: ${path}" >&2
    echo "expected=${expected}" >&2
    echo "actual=${actual}" >&2
    return 1
  }
  echo "hash PASS: ${path}"
}

[[ -x "${PYTHON_BIN}" ]] || { echo "missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${MODEL}" ]] || { echo "missing model directory: ${MODEL}" >&2; exit 1; }

shard_count=$(find -L "${MODEL}" -maxdepth 1 -type f -name 'model-*-of-00016.safetensors' | wc -l)
shard_bytes=$(find -L "${MODEL}" -maxdepth 1 -type f -name 'model-*-of-00016.safetensors' -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')
[[ "${shard_count}" -eq 16 ]] || {
  echo "model has ${shard_count}/16 final shards" >&2
  exit 1
}
[[ "${shard_bytes}" -eq 61066575656 ]] || {
  echo "model shard byte total mismatch: ${shard_bytes}" >&2
  exit 1
}
[[ $(stat -Lc '%s' "${GGUF}") -eq 32483930560 ]] || {
  echo "GGUF size mismatch: ${GGUF}" >&2
  exit 1
}

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import platform
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python mismatch: expected=3.10.x actual={platform.python_version()}")
print(f"Python PASS: {platform.python_version()} (validated reference: 3.10.14)")
if platform.machine() != "x86_64":
    raise SystemExit(f"architecture mismatch: expected=x86_64 actual={platform.machine()}")
expected = {
    "sglang-kt": "0.6.3",
    "kt-kernel": "0.5.0",
    "torch": "2.9.1",
    "transformers": "5.12.1",
    "numpy": "2.2.6",
}
for package, version in expected.items():
    actual = md.version(package)
    if actual != version:
        raise SystemExit(f"version mismatch: {package} expected={version} actual={actual}")
    print(f"version PASS: {package}=={actual}")
PY

KT_ROOT=$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
from pathlib import Path

print(Path(md.distribution("kt-kernel").locate_file("kt_kernel")).resolve())
PY
)
check_hash "${KT_ROOT}/__init__.py" 2a2e1304a7dc054427008463af28440647d92f0b3a95c51626b86819c1841df2
check_hash "${KT_ROOT}/experts.py" c6e6390d1543b3d8c88fcfbe17314b001f96032349e7efaa25a7ef30fd463c88
check_hash "${KT_ROOT}/experts_base.py" ab022f6a862a63e179c45861f7e8ddc07b5818820c23aecf48453f566770ec59
check_hash "${KT_ROOT}/utils/llamafile.py" 80b2b1ee502a4fa6f45029da68d8514e286e247ca8dd1d5721f0d195073dd5d8
check_hash "${KT_ROOT}/_kt_kernel_ext_avx2.cpython-310-x86_64-linux-gnu.so" 066940ea0a97ab98bcfd369358a2be0178772381d3690b9870c679e963758d1f
if [[ -n "${hwloc_path:-}" ]]; then
  check_hash "${hwloc_path}" 2efe6e32d0d3454b2bce0d3b8cd437f48691475717be4b431f96bcc2d4d9bc54
fi

check_hash "${MODEL}/config.json" a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438
check_hash "${MODEL}/model.safetensors.index.json" 8dde190b862c7c80ec7403c6495de00c60bbaf246ed479cee4506284989c584c
check_hash "${MODEL}/tokenizer.json" aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
check_hash "${MODEL}/tokenizer_config.json" a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3
check_hash "${WORKLOAD}" 1b7ea6af331824b77c84a85840beac7bbb412ac12d7788958cf39b401d835b32
check_hash "${IDENTITY}" efe072e06c6f9b8cdddd835971f774f98d7330b00d88a11f4e82bb6ab0e2d234
check_hash "${ORACLE}" 6ec7591cac1b5274716bc50e6c65faa361655ae6dc7fed6d6c3be24da376dfb1
check_hash "${STATS}" 493e5e4976a980f34497c56da4e2ba044d44a8d83b46c96027b12364fa81fd38

"${PYTHON_BIN}" -c 'import kt_kernel; print(f"kt-kernel import: PASS ({kt_kernel.__cpu_variant__})")'
echo "external environment verification: PASS"
