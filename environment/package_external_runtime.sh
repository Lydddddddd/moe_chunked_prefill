#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${ROOT}/.venv_kt/bin/python"}
if [[ ! -x "${PYTHON_BIN}" && -x "${ROOT}/../.venv_kt/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/../.venv_kt/bin/python"
fi
[[ -x "${PYTHON_BIN}" ]] || { echo "missing Python: ${PYTHON_BIN}" >&2; exit 1; }
OUTPUT=${OUTPUT:-"${ROOT}/reproduction/m20_external_runtime_cp310_x86_64.tar.zst"}
KT_ROOT=$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
from pathlib import Path

print(Path(md.distribution("kt-kernel").locate_file("kt_kernel")).resolve())
PY
)
binary_name="_kt_kernel_ext_avx2.cpython-310-x86_64-linux-gnu.so"
KT_AVX2_BINARY=${KT_AVX2_BINARY:-"${KT_ROOT}/${binary_name}"}
HWLOC_LIB=${HWLOC_LIB:-"${ROOT}/../third_party/hwloc_dev/root/usr/lib/x86_64-linux-gnu/libhwloc.so.15.5.2"}

[[ -f "${KT_AVX2_BINARY}" ]] || { echo "missing AVX2 binary: ${KT_AVX2_BINARY}" >&2; exit 1; }
[[ -f "${HWLOC_LIB}" ]] || { echo "missing hwloc library: ${HWLOC_LIB}" >&2; exit 1; }
[[ $(sha256sum "${KT_AVX2_BINARY}" | awk '{print $1}') == "066940ea0a97ab98bcfd369358a2be0178772381d3690b9870c679e963758d1f" ]] || {
  echo "AVX2 binary is not the approved M20 build" >&2
  exit 1
}
[[ $(sha256sum "${HWLOC_LIB}" | awk '{print $1}') == "2efe6e32d0d3454b2bce0d3b8cd437f48691475717be4b431f96bcc2d4d9bc54" ]] || {
  echo "hwloc library is not the validated build" >&2
  exit 1
}

tmp_dir=$(mktemp -d /tmp/m20-external-runtime.XXXXXX)
trap 'rm -rf -- "${tmp_dir}"' EXIT
bundle="${tmp_dir}/external_runtime"
mkdir -p "${bundle}/kt_kernel" "${bundle}/lib" "$(dirname -- "${OUTPUT}")"
install -m 0755 "${KT_AVX2_BINARY}" "${bundle}/kt_kernel/${binary_name}"
install -m 0755 "${HWLOC_LIB}" "${bundle}/lib/libhwloc.so.15.5.2"
ln -s libhwloc.so.15.5.2 "${bundle}/lib/libhwloc.so.15"
(
  cd "${bundle}"
  sha256sum "kt_kernel/${binary_name}" "lib/libhwloc.so.15.5.2" > SHA256SUMS
)
tar -I 'zstd -T0 -10' -cf "${OUTPUT}" -C "${tmp_dir}" external_runtime
(
  cd "$(dirname -- "${OUTPUT}")"
  sha256sum "$(basename -- "${OUTPUT}")" > "$(basename -- "${OUTPUT}").sha256"
)
echo "external runtime bundle: ${OUTPUT}"
echo "external runtime checksum: ${OUTPUT}.sha256"
