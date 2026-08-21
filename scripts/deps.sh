#!/usr/bin/env bash
# Provision only the dependencies requested by a build profile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
DEPS_ROOT="${AI_CORE_DEPS_ROOT:-${ROOT}/.deps/${HOST_OS}_${HOST_ARCH}}"
ORT_VERSION="${AI_CORE_ONNXRUNTIME_VERSION:-1.20.1}"

log() { printf '[deps] %s\n' "$*"; }
die() { printf '[deps] error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  scripts/deps.sh list
  scripts/deps.sh init <profile> [profile ...]

Profiles:
  core          no third-party dependency
  config        nlohmann/json source
  vision        OpenCV source (for BUNDLED provider)
  onnxruntime   ONNX Runtime SDK under .deps/<platform>
  ncnn          NCNN SDK from $AI_CORE_NCNN_ARCHIVE
  tensorrt      TensorRT SDK from $AI_CORE_TENSORRT_ARCHIVE
  decryption    encryption-tool source and its nested submodules
  testing       GoogleTest source
  benchmarking  Google Benchmark source
  developer     config + vision + onnxruntime + testing
EOF
}

init_submodule() {
  local path="$1"
  log "initializing ${path}"
  git -C "${ROOT}" submodule update --init --depth 1 -- "${path}"
}

init_recursive_submodule() {
  local path="$1"
  log "initializing ${path} recursively"
  git -C "${ROOT}" submodule update --init --recursive --depth 1 -- "${path}"
}

install_archive() {
  local dependency="$1"
  local archive="$2"
  local destination="$3"

  [[ -f "${archive}" ]] || die "${dependency} archive not found: ${archive}"
  [[ ! -e "${destination}" ]] || {
    log "${dependency} already present (${destination})"
    return
  }

  local staging
  staging="$(mktemp -d)"
  trap 'rm -rf -- "${staging}"' RETURN
  mkdir -p "${staging}/unpack" "$(dirname "${destination}")"
  tar -xf "${archive}" -C "${staging}/unpack"

  local entries
  entries=("${staging}/unpack"/*)
  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    mv -- "${entries[0]}" "${destination}"
  else
    mkdir -p "${staging}/payload"
    mv -- "${entries[@]}" "${staging}/payload/"
    mv -- "${staging}/payload" "${destination}"
  fi
  trap - RETURN
  rm -rf -- "${staging}"
  log "${dependency} installed (${destination})"
}

init_onnxruntime() {
  local destination="${DEPS_ROOT}/onnxruntime"
  if [[ -f "${destination}/include/onnxruntime_cxx_api.h" ||
        -f "${destination}/include/onnxruntime/onnxruntime_cxx_api.h" ]]; then
    log "ONNX Runtime already present (${destination})"
    return
  fi

  [[ "${HOST_OS}_${HOST_ARCH}" == "Linux_x86_64" ]] ||
    die "automatic ONNX Runtime provisioning currently supports Linux_x86_64; set AI_CORE_ONNXRUNTIME_ROOT manually"
  command -v curl >/dev/null || die "curl is required for ONNX Runtime"

  local archive="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
  local url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${archive}"
  local staging
  staging="$(mktemp -d)"
  trap 'rm -rf -- "${staging}"' RETURN
  log "downloading ONNX Runtime ${ORT_VERSION}"
  curl -fL "${url}" -o "${staging}/${archive}"
  install_archive "ONNX Runtime" "${staging}/${archive}" "${destination}"
  trap - RETURN
  rm -rf -- "${staging}"

  local cmake_dir="${destination}/lib/cmake/onnxruntime"
  if [[ -d "${cmake_dir}" ]]; then
    [[ ! -f "${cmake_dir}/onnxruntimeTargets-release.cmake" ]] ||
      sed -i 's#/lib64/#/lib/#g' \
        "${cmake_dir}/onnxruntimeTargets-release.cmake"
    [[ ! -f "${cmake_dir}/onnxruntimeTargets.cmake" ]] ||
      sed -i 's#/include/onnxruntime"#/include"#g' \
        "${cmake_dir}/onnxruntimeTargets.cmake"
  fi
  [[ -f "${destination}/include/onnxruntime_cxx_api.h" ||
     -f "${destination}/include/onnxruntime/onnxruntime_cxx_api.h" ]] ||
    die "ONNX Runtime archive has an unexpected layout"
}

init_profile() {
  case "$1" in
    core) ;;
    config) init_submodule third_party/config/nlohmann_json ;;
    vision) init_submodule third_party/plugins/opencv ;;
    onnxruntime) init_onnxruntime ;;
    ncnn)
      [[ -d "${DEPS_ROOT}/ncnn" ]] ||
        install_archive "NCNN" "${AI_CORE_NCNN_ARCHIVE:-}" "${DEPS_ROOT}/ncnn"
      ;;
    tensorrt)
      [[ -d "${DEPS_ROOT}/tensorrt" ]] ||
        install_archive "TensorRT" "${AI_CORE_TENSORRT_ARCHIVE:-}" "${DEPS_ROOT}/tensorrt"
      ;;
    decryption)
      init_recursive_submodule third_party/plugins/encryption_tool
      ;;
    testing) init_submodule third_party/testing/googletest ;;
    benchmarking)
      init_submodule third_party/benchmarking/google_benchmark
      ;;
    developer)
      init_profile config
      init_profile vision
      init_profile onnxruntime
      init_profile testing
      ;;
    *) die "unknown dependency profile: $1 (run: scripts/deps.sh list)" ;;
  esac
}

case "${1:-}" in
  list) usage ;;
  init)
    shift
    [[ $# -gt 0 ]] || die "at least one profile is required"
    mkdir -p "${DEPS_ROOT}"
    for profile in "$@"; do
      init_profile "${profile}"
    done
    log "dependency root: ${DEPS_ROOT}"
    ;;
  -h|--help|'') usage ;;
  *) die "unknown command: $1 (run: scripts/deps.sh --help)" ;;
esac
