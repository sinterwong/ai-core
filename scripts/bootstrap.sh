#!/usr/bin/env bash
# Reproducible developer build using the same dependency profiles as CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${AI_CORE_BUILD_DIR:-${ROOT}/build}"
INSTALL_DIR="${AI_CORE_INSTALL_DIR:-${ROOT}/install}"
DEPS_ROOT="${AI_CORE_DEPS_ROOT:-${ROOT}/.deps/$(uname -s)_$(uname -m)}"
OPENCV_PROVIDER="${AI_CORE_OPENCV_PROVIDER:-BUNDLED}"
JOBS="${AI_CORE_JOBS:-$(nproc)}"
RUN_TESTS=ON
BUILD_BENCHMARKS=OFF
BUILD_NCNN=OFF
BUILD_TENSORRT=OFF
ENABLE_DECRYPTION=OFF

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [options]
  --opencv-provider BUNDLED|SYSTEM
  --with-ncnn
  --with-tensorrt
  --with-decryption
  --benchmarks
  --no-tests
  --jobs N
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --opencv-provider) OPENCV_PROVIDER="$2"; shift ;;
    --with-ncnn) BUILD_NCNN=ON ;;
    --with-tensorrt) BUILD_TENSORRT=ON ;;
    --with-decryption) ENABLE_DECRYPTION=ON ;;
    --benchmarks) BUILD_BENCHMARKS=ON ;;
    --no-tests) RUN_TESTS=OFF ;;
    --jobs) JOBS="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'bootstrap: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

case "${OPENCV_PROVIDER}" in
  BUNDLED) "${ROOT}/scripts/deps.sh" init vision ;;
  SYSTEM) ;;
  *) printf 'bootstrap: invalid OpenCV provider: %s\n' "${OPENCV_PROVIDER}" >&2; exit 2 ;;
esac

profiles=(config onnxruntime)
[[ "${RUN_TESTS}" == ON ]] && profiles+=(testing)
[[ "${BUILD_BENCHMARKS}" == ON ]] && profiles+=(benchmarking)
[[ "${BUILD_NCNN}" == ON ]] && profiles+=(ncnn)
[[ "${BUILD_TENSORRT}" == ON ]] && profiles+=(tensorrt)
[[ "${ENABLE_DECRYPTION}" == ON ]] && profiles+=(decryption)
"${ROOT}/scripts/deps.sh" init "${profiles[@]}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DAI_CORE_DEPS_ROOT="${DEPS_ROOT}" \
  -DAI_CORE_OPENCV_PROVIDER="${OPENCV_PROVIDER}" \
  -DAI_CORE_BUILD_CONFIG=ON \
  -DAI_CORE_BUILD_PLUGIN_PREPROC_OPENCV=ON \
  -DAI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV=ON \
  -DAI_CORE_BUILD_PLUGIN_ONNXRUNTIME=ON \
  -DAI_CORE_BUILD_PLUGIN_NCNN="${BUILD_NCNN}" \
  -DAI_CORE_BUILD_PLUGIN_TENSORRT="${BUILD_TENSORRT}" \
  -DAI_CORE_ENABLE_MODEL_DECRYPTION="${ENABLE_DECRYPTION}" \
  -DAI_CORE_BUILD_TESTS="${RUN_TESTS}" \
  -DAI_CORE_BUILD_BENCHMARKS="${BUILD_BENCHMARKS}"

cmake --build "${BUILD_DIR}" -j"${JOBS}"
cmake --install "${BUILD_DIR}"
if [[ "${RUN_TESTS}" == ON ]]; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

printf 'ai-core installed to %s\n' "${INSTALL_DIR}"
