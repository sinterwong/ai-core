#!/usr/bin/env bash
# Provision the model assets the integration tests need, reproducibly.
#
#   - Base models (ONNX, NCNN) are the source of truth. If missing, they are
#     downloaded from $AI_CORE_MODEL_BASE_URL (a directory URL; each file is
#     fetched by name).
#   - Derived, machine-specific artifacts (TensorRT .engine files) are built
#     locally from the ONNX models with trtexec, so they are never committed.
#
# Usage:
#   scripts/fetch_models.sh              # fetch base models + build TRT engines
#   scripts/fetch_models.sh --base-only  # only ensure base models are present
#   scripts/fetch_models.sh --trt-only   # only (re)build TRT engines
#
# Env:
#   AI_CORE_MODEL_BASE_URL   directory URL to download base models from
#   TRTEXEC                  path to trtexec (default: search PATH)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT}/assets/models"
TRTEXEC="${TRTEXEC:-$(command -v trtexec || true)}"

# Base models: filename -> required (always) . Committed NCNN params/bin and
# the OCR/reco onnx are listed so a fresh checkout can be validated.
BASE_MODELS=(
  yolov11n.onnx
  yolov11n-fp16.onnx
  yolov11n.ncnn.param
  yolov11n.ncnn.bin
  ch_PP_ocr_det.onnx
  cnocr136fc.onnx
)

# TensorRT engines: "output_engine:source_onnx:extra_trtexec_flags".
# cnocr has a dynamic-width input 'x' [1,1,32,W]; the profile must cover the
# integration test's 1x1x32x128 shape.
TRT_ENGINES=(
  "yolov11n_trt_fp16.engine:yolov11n.onnx:--fp16"
  "cnocr136fc_fp16_dynamic.engine:cnocr136fc.onnx:--fp16 --minShapes=x:1x1x32x32 --optShapes=x:1x1x32x128 --maxShapes=x:1x1x32x512"
)

log() { printf '\033[36m[fetch_models]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[fetch_models] ERROR:\033[0m %s\n' "$*" >&2; }

ensure_base_models() {
  mkdir -p "${MODELS_DIR}"
  local missing=()
  for m in "${BASE_MODELS[@]}"; do
    [[ -f "${MODELS_DIR}/${m}" ]] || missing+=("${m}")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    log "all base models present"
    return
  fi
  if [[ -z "${AI_CORE_MODEL_BASE_URL:-}" ]]; then
    err "missing base models: ${missing[*]}"
    err "set AI_CORE_MODEL_BASE_URL to a directory URL to download them."
    exit 1
  fi
  for m in "${missing[@]}"; do
    log "downloading ${m}"
    wget -q "${AI_CORE_MODEL_BASE_URL%/}/${m}" -O "${MODELS_DIR}/${m}"
  done
}

build_trt_engines() {
  if [[ -z "${TRTEXEC}" ]]; then
    log "trtexec not found; skipping TensorRT engine build (TRT tests will skip)"
    return
  fi
  for spec in "${TRT_ENGINES[@]}"; do
    IFS=':' read -r engine onnx flags <<<"${spec}"
    local out="${MODELS_DIR}/${engine}"
    local src="${MODELS_DIR}/${onnx}"
    if [[ ! -f "${src}" ]]; then
      err "source onnx ${onnx} missing for ${engine}; run base fetch first"
      exit 1
    fi
    if [[ -f "${out}" && "${out}" -nt "${src}" ]]; then
      log "${engine} up to date"
      continue
    fi
    log "building ${engine} from ${onnx} (${flags})"
    "${TRTEXEC}" --onnx="${src}" ${flags} --saveEngine="${out}" >/dev/null 2>&1 \
      || { err "trtexec failed for ${engine}"; exit 1; }
  done
}

MODE="${1:-all}"
case "${MODE}" in
  --base-only) ensure_base_models ;;
  --trt-only)  build_trt_engines ;;
  all|"")      ensure_base_models; build_trt_engines ;;
  *) err "unknown option: ${MODE}"; exit 2 ;;
esac
log "done"
