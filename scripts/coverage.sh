#!/usr/bin/env bash
# Measure the dependency-free core suite. Usage: coverage.sh [threshold].
set -euo pipefail

THRESHOLD="${1:-80}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${AI_CORE_COVERAGE_BUILD_DIR:-${ROOT}/build-cov}"
GCOV="${GCOV:-gcov}"

# Keep the measured sources aligned with the dependency-free tests selected
# below. Plugin orchestration and backend code are covered by their integration
# suites and must not dilute this core-only coverage gate.
CORE_FILTERS=(
  --filter 'include/ai_core/algo_types\.hpp'
  --filter 'include/ai_core/common_types\.hpp'
  --filter 'include/ai_core/data_packet\.hpp'
  --filter 'include/ai_core/error_code\.hpp'
  --filter 'include/ai_core/param_center\.hpp'
  --filter 'include/ai_core/tensor_data\.hpp'
  --filter 'include/ai_core/type_safe_factory\.hpp'
  --filter 'include/ai_core/typed_buffer\.hpp'
  --filter 'src/typed_buffer\.cpp'
)

"${ROOT}/scripts/deps.sh" init testing
cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAI_CORE_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS=--coverage \
  -DCMAKE_SHARED_LINKER_FLAGS=--coverage
cmake --build "${BUILD_DIR}" -j"$(nproc)"

find "${BUILD_DIR}" -name '*.gcda' -delete
ctest --test-dir "${BUILD_DIR}" --output-on-failure -L core

python3 -m gcovr "${BUILD_DIR}" --root "${ROOT}" \
  --gcov-executable "${GCOV}" \
  --gcov-ignore-parse-errors=negative_hits.warn \
  "${CORE_FILTERS[@]}" \
  --fail-under-line "${THRESHOLD}" \
  --print-summary
