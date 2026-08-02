#!/usr/bin/env bash
# One command from a fresh clone to a green build: dependencies -> configure
# -> build -> install -> provision models -> test.
#
# Usage:
#   scripts/bootstrap.sh [--with-trt] [--with-ncnn] [--no-test] [--jobs N]
#
# Dependencies follow the same recipe as CI (.github/workflows/ci.yml):
#   - ONNX Runtime from the official Microsoft release
#   - OpenCV from the system (apt: libopencv-dev)
# Deliberately not a single vendored bundle: shipping our own OpenCV next to the
# system one puts two libopencv_core.so in one process, and passing a cv::Mat
# across that boundary is undefined behaviour with very indirect symptoms.
#
# Env:
#   ORT_VERSION       ONNX Runtime version to fetch (default below).
#   TRTEXEC           path to trtexec (for TensorRT engine provisioning).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WITH_TRT=OFF
WITH_NCNN=OFF
RUN_TESTS=1
JOBS="$(nproc)"

# Single source of truth for the ORT version. Keep in sync with
# .github/workflows/ci.yml (env.ORT_VERSION).
ORT_VERSION="${ORT_VERSION:-1.20.1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-trt)  WITH_TRT=ON ;;
    --with-ncnn) WITH_NCNN=ON ;;
    --no-ncnn)   WITH_NCNN=OFF ;;
    --no-test)   RUN_TESTS=0 ;;
    --jobs)      JOBS="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '\033[36m[bootstrap]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

TARGET_DIR="3rdparty/target/Linux_x86_64"
ORT_HOME="${TARGET_DIR}/onnxruntime"

# --- 1. Third-party dependencies --------------------------------------------
command -v curl >/dev/null || die "curl is required"
pkg-config --exists opencv4 2>/dev/null || \
  log "note: system OpenCV not found via pkg-config; install libopencv-dev if configure fails"

if [[ -f "${ORT_HOME}/lib/libonnxruntime.so" ]]; then
  log "ONNX Runtime present (${ORT_HOME})"
else
  ORT_TARBALL="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
  ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}"
  log "downloading ONNX Runtime ${ORT_VERSION}"
  # -f so a 404 fails here with a readable message, instead of surfacing later
  # as "tar: not in gzip format" when a 9-byte error page gets unpacked.
  curl -fL "${ORT_URL}" -o "/tmp/${ORT_TARBALL}" \
    || die "failed to download ${ORT_URL}"
  mkdir -p "${TARGET_DIR}"
  rm -rf "${ORT_HOME}"
  tar -xzf "/tmp/${ORT_TARBALL}" -C /tmp
  mv "/tmp/onnxruntime-linux-x64-${ORT_VERSION}" "${ORT_HOME}"
  rm -f "/tmp/${ORT_TARBALL}"
fi

# The 1.20.x release tarball ships libs in lib/ and headers in include/, but its
# own cmake export references lib64/ and include/onnxruntime. Without this,
# find_package(onnxruntime) resolves paths that do not exist. Idempotent.
ORT_CMAKE="${ORT_HOME}/lib/cmake/onnxruntime"
if [[ -d "${ORT_CMAKE}" ]]; then
  sed -i 's#/lib64/#/lib/#g' "${ORT_CMAKE}/onnxruntimeTargets-release.cmake"
  sed -i 's#/include/onnxruntime"#/include"#g' "${ORT_CMAKE}/onnxruntimeTargets.cmake"
fi

[[ -f "${ORT_HOME}/include/onnxruntime_cxx_api.h" ]] \
  || die "missing ${ORT_HOME}/include/onnxruntime_cxx_api.h"
[[ -f "${ORT_HOME}/lib/libonnxruntime.so" ]] \
  || die "missing ${ORT_HOME}/lib/libonnxruntime.so"
log "ONNX Runtime ready (${ORT_HOME})"

# --- 2. Configure + build + install -----------------------------------------
log "configuring (TRT=${WITH_TRT} NCNN=${WITH_NCNN})"
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${ROOT}/install" \
  -DBUILD_AI_CORE_TESTS=ON \
  -DBUILD_AI_CORE_EXAMPLES=ON \
  -DBUILD_AI_CORE_CONFIG=ON \
  -DWITH_ORT_ENGINE=ON \
  -DWITH_NCNN_ENGINE="${WITH_NCNN}" \
  -DWITH_TRT_ENGINE="${WITH_TRT}"

log "building (-j${JOBS})"
cmake --build build -j"${JOBS}"
cmake --install build >/dev/null
log "installed to ${ROOT}/install"

# Exactly one OpenCV must be linked in; two would be the UB described above.
OPENCV_COUNT="$(ldd "${ROOT}/install/lib/libai_core.so" 2>/dev/null \
  | grep -c 'libopencv_core' || true)"
if [[ "${OPENCV_COUNT}" != "1" ]]; then
  die "expected exactly 1 libopencv_core in libai_core.so, found ${OPENCV_COUNT}"
fi

# --- 3. Provision models -----------------------------------------------------
if [[ "${WITH_TRT}" == "ON" ]]; then
  log "provisioning models (incl. TensorRT engines)"
  scripts/fetch_models.sh || log "model provisioning reported issues (continuing)"
else
  scripts/fetch_models.sh --base-only || true
fi

# --- 4. Test -----------------------------------------------------------------
if [[ "${RUN_TESTS}" == "1" ]]; then
  log "running tests"
  LIBS="${ROOT}/install/lib:$(ls -d "${ROOT}"/${TARGET_DIR}/*/lib 2>/dev/null | tr '\n' ':')"
  ( cd "${ROOT}/install" && LD_LIBRARY_PATH="${LIBS}" ./tests/ai_core_tests )
  log "all tests passed"
fi

log "done. Install tree: ${ROOT}/install"
