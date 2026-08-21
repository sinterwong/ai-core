#!/usr/bin/env bash
# Measure line coverage of the backend-agnostic core components and enforce a
# minimum threshold. Runs the asset-free unit tests only, so it needs no model
# downloads and no GPU. Intended for CI (ORT-only build) and local use.
#
# Usage: scripts/coverage.sh [threshold_percent]   (default 80)
set -euo pipefail

THRESHOLD="${1:-80}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build-cov"
INSTALL_DIR="${ROOT}/install-cov"
GCOV="${GCOV:-gcov}"

# Core components: data types + decode logic. Backend engines (ort/ncnn/trt)
# are integration-tested, not counted here.
CORE_FILTERS=(
  --filter 'include/ai_core/typed_buffer\.hpp'
  --filter 'include/ai_core/tensor_data\.hpp'
  --filter 'include/ai_core/data_packet\.hpp'
  --filter 'include/ai_core/param_center\.hpp'
  --filter 'include/ai_core/type_safe_factory\.hpp'
  --filter 'include/ai_core/image_view\.hpp'
  --filter 'include/ai_core/opencv_interop\.hpp'
  --filter 'include/ai_core/common_types\.hpp'
  --filter 'src/typed_buffer\.cpp'
  --filter 'plugins/common/opencv/vision_util\.cpp'
  --filter 'src/param_validation\.hpp'
  --filter 'plugins/preproc/opencv/preproc/cpu_generic_preprocessor\.cpp'
  --filter 'plugins/preproc/opencv/preproc/frame_with_mask_prep\.cpp'
  --filter 'plugins/preproc/opencv/preproc/generic_frame_preproc_base\.cpp'
  --filter 'plugins/postproc/opencv/postproc/.*\.cpp'
)

# Unit-test suites only (no model assets, no crashing integration teardown).
UNIT_FILTER='TypedBufferTest.*:TensorDataTest.*:DataPacketTest.*:ParamCenterTest.*:FactoryTest.*:ImageViewTest.*:InteropTest.*:GeometryTest.*:CpuPreprocTest.*:FrameWithMaskTest.*:ParamBindingTest.*:*Decode*:EscaleResizeWithPadTest.*:CoordinateRestorationTest.*:ScaleRatioTest.*:NmsTest.*:PlanPixelFormatTest.*:ConvertPixelFormatTest.*'

echo "== Configuring instrumented build =="
cmake -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DBUILD_AI_CORE_TESTS=ON \
  -DAI_CORE_BUILD_BUNDLED_PLUGINS=ON \
  -DWITH_ORT_ENGINE=ON -DWITH_NCNN_ENGINE=OFF -DWITH_TRT_ENGINE=OFF \
  -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
  -DCMAKE_SHARED_LINKER_FLAGS="--coverage"

echo "== Building =="
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}" >/dev/null

echo "== Running unit tests =="
find "${BUILD_DIR}" -name '*.gcda' -delete
LIBS="${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/ai_core/plugins:$(ls -d "${ROOT}"/3rdparty/target/Linux_x86_64/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
( cd "${INSTALL_DIR}" && LD_LIBRARY_PATH="${LIBS}" \
    ./tests/ai_core_tests --gtest_filter="${UNIT_FILTER}" )

echo "== Core-component coverage (threshold ${THRESHOLD}%) =="
python3 -m gcovr "${BUILD_DIR}" --root "${ROOT}" \
  --gcov-executable "${GCOV}" \
  --gcov-ignore-parse-errors=negative_hits.warn \
  "${CORE_FILTERS[@]}" \
  --fail-under-line "${THRESHOLD}" \
  --print-summary
