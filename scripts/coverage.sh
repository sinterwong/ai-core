#!/usr/bin/env bash
# Measure the dependency-free core suite. Usage: coverage.sh [threshold].
set -euo pipefail

THRESHOLD="${1:-80}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${AI_CORE_COVERAGE_BUILD_DIR:-${ROOT}/build-cov}"
GCOV="${GCOV:-gcov}"

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
  --filter 'include/ai_core/.*\.hpp' \
  --filter 'src/.*\.(cpp|hpp)' \
  --exclude 'src/config/.*' \
  --fail-under-line "${THRESHOLD}" \
  --print-summary
