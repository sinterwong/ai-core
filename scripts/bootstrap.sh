#!/usr/bin/env bash
# One command from a fresh clone to a green build: dependencies -> configure
# -> build -> install -> provision models -> test.
#
# Usage:
#   scripts/bootstrap.sh [--with-trt] [--with-ncnn] [--no-test] [--jobs N]
#
# Env:
#   AI_CORE_DEP_URL   URL to the prebuilt 3rdparty dependency tarball
#                     (defaults to the published release).
#   TRTEXEC           path to trtexec (for TensorRT engine provisioning).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WITH_TRT=OFF
WITH_NCNN=ON
RUN_TESTS=1
JOBS="$(nproc)"
DEP_URL="${AI_CORE_DEP_URL:-https://github.com/sinterwong/ai-core/releases/download/v1.1.1-alpha/dependency-Linux_x86_64.tgz}"

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

TARGET_DIR="3rdparty/target/Linux_x86_64"

# --- 1. Third-party dependencies --------------------------------------------
if [[ -d "${TARGET_DIR}/opencv" ]]; then
  log "third-party deps present (${TARGET_DIR})"
else
  log "downloading prebuilt dependencies from ${DEP_URL}"
  mkdir -p 3rdparty/target
  curl -L "${DEP_URL}" -o /tmp/ai_core_dep.tgz
  tar -xzf /tmp/ai_core_dep.tgz -C 3rdparty/target/
fi

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
