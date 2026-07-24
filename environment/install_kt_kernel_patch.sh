#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${ROOT}/.venv_kt/bin/python"}
if [[ ! -x "${PYTHON_BIN}" && -x "${ROOT}/../.venv_kt/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/../.venv_kt/bin/python"
fi
EXTERNAL_RUNTIME_DIR=${EXTERNAL_RUNTIME_DIR:-"${ROOT}/environment/external_runtime"}
PATCH_FILE="${ROOT}/environment/kt_kernel_patch/kt_kernel_0.5.0_m20.patch"

[[ -x "${PYTHON_BIN}" ]] || { echo "missing Python: ${PYTHON_BIN}" >&2; exit 1; }
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"kt-kernel binary requires CPython 3.10, got {sys.version.split()[0]}")
if md.version("kt-kernel") != "0.5.0":
    raise SystemExit(f"kt-kernel==0.5.0 required, got {md.version('kt-kernel')}")
PY

SITE_ROOT=$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
from pathlib import Path

print(Path(md.distribution("kt-kernel").locate_file("")).resolve())
PY
)
KT_ROOT="${SITE_ROOT}/kt_kernel"

relative_sources=(
  "__init__.py"
  "experts.py"
  "experts_base.py"
  "utils/llamafile.py"
)
base_hashes=(
  "d970d6780072685bbcb0534d6a881671256d5744b3de7422c2ae8e6a5e337d1a"
  "7aa5356b52bbb227a139a4600b1701948bdd18f20fa4700445541b5ac826acc3"
  "25be47f29934a2e7eaf9fed6c2c0bc1261644550ff2d40a37d8751ba5757185e"
  "0ddc6b90fc3c741fe074b4f15a53b1f6c1b94ebd0d661c7c4436446a544e745a"
)
patched_hashes=(
  "2a2e1304a7dc054427008463af28440647d92f0b3a95c51626b86819c1841df2"
  "c6e6390d1543b3d8c88fcfbe17314b001f96032349e7efaa25a7ef30fd463c88"
  "ab022f6a862a63e179c45861f7e8ddc07b5818820c23aecf48453f566770ec59"
  "80b2b1ee502a4fa6f45029da68d8514e286e247ca8dd1d5721f0d195073dd5d8"
)
binary_name="_kt_kernel_ext_avx2.cpython-310-x86_64-linux-gnu.so"
binary_target="${KT_ROOT}/${binary_name}"
binary_source=${KT_AVX2_BINARY:-"${EXTERNAL_RUNTIME_DIR}/kt_kernel/${binary_name}"}
binary_hash="066940ea0a97ab98bcfd369358a2be0178772381d3690b9870c679e963758d1f"
current_binary_hash=$(sha256sum "${binary_target}" | awk '{print $1}')
install_binary=0
if [[ "${current_binary_hash}" != "${binary_hash}" ]]; then
  [[ -f "${binary_source}" ]] || {
    echo "missing approved AVX2 binary: ${binary_source}" >&2
    echo "set EXTERNAL_RUNTIME_DIR or KT_AVX2_BINARY to the delivered runtime bundle" >&2
    exit 1
  }
  source_hash=$(sha256sum "${binary_source}" | awk '{print $1}')
  [[ "${source_hash}" == "${binary_hash}" ]] || {
    echo "AVX2 binary hash mismatch: ${binary_source}" >&2
    exit 1
  }
  install_binary=1
fi

all_patched=1
for index in "${!relative_sources[@]}"; do
  actual=$(sha256sum "${KT_ROOT}/${relative_sources[$index]}" | awk '{print $1}')
  [[ "${actual}" == "${patched_hashes[$index]}" ]] || all_patched=0
done

if ((all_patched == 0)); then
  for index in "${!relative_sources[@]}"; do
    path="${KT_ROOT}/${relative_sources[$index]}"
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ "${actual}" != "${base_hashes[$index]}" ]]; then
      echo "refusing to patch unexpected kt-kernel source: ${path}" >&2
      echo "expected base=${base_hashes[$index]} actual=${actual}" >&2
      exit 1
    fi
  done
  patch --batch --forward -p1 -d "${SITE_ROOT}" < "${PATCH_FILE}"
fi

if ((install_binary == 1)); then
  install -m 0755 "${binary_source}" "${binary_target}"
fi

for index in "${!relative_sources[@]}"; do
  echo "${patched_hashes[$index]}  ${KT_ROOT}/${relative_sources[$index]}"
done | sha256sum --check --status || {
  echo "installed kt-kernel source patch failed hash verification" >&2
  exit 1
}
echo "${binary_hash}  ${binary_target}" | sha256sum --check --status || {
  echo "installed AVX2 binary failed hash verification" >&2
  exit 1
}

if [[ -d "${EXTERNAL_RUNTIME_DIR}/lib" ]]; then
  hwloc_path="${EXTERNAL_RUNTIME_DIR}/lib/libhwloc.so.15.5.2"
  [[ -f "${hwloc_path}" ]] || { echo "missing hwloc library: ${hwloc_path}" >&2; exit 1; }
  [[ $(sha256sum "${hwloc_path}" | awk '{print $1}') == "2efe6e32d0d3454b2bce0d3b8cd437f48691475717be4b431f96bcc2d4d9bc54" ]] || {
    echo "hwloc library hash mismatch: ${hwloc_path}" >&2
    exit 1
  }
  export LD_LIBRARY_PATH="${EXTERNAL_RUNTIME_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi
"${PYTHON_BIN}" -c 'import kt_kernel; print(f"kt-kernel import: PASS ({kt_kernel.__cpu_variant__})")'
echo "kt-kernel M20 patch install: PASS (${KT_ROOT})"
