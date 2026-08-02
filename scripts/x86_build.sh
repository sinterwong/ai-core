#!/bin/bash

# Clean rebuild of ai-core for x86_64, including benchmarks.
#
# Dependency provisioning lives in scripts/bootstrap.sh — this script only adds
# the "wipe everything first" and "also run benchmarks" parts. Keeping one
# recipe for dependencies is deliberate: the three copies that used to exist
# drifted onto three different release tags, two of which 404'd.

# Exit immediately if a command exits with a non-zero status.
set -e

REPO_DIR=$(pwd)

# Define the build directory
BUILD_DIR="${REPO_DIR}/build"
INSTALL_DIR="${REPO_DIR}/install"

# Clean up previous build artifacts
if [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning up previous build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

if [ -d "${INSTALL_DIR}" ]; then
    echo "Cleaning up previous install directory: ${INSTALL_DIR}"
    rm -rf "${INSTALL_DIR}"
fi

echo "Updating git submodules..."
git submodule update --init --recursive

echo "Updating apt packages and installing build tools..."
sudo apt-get update
sudo apt-get install -y gcc-13 g++-13
sudo apt-get install -y ninja-build libopencv-dev

echo "Provisioning dependencies, building, installing and testing..."
scripts/bootstrap.sh

echo "Building benchmarks..."
cmake -B "${BUILD_DIR}" -DBUILD_AI_CORE_BENCHMARKS=ON
cmake --build "${BUILD_DIR}" --config Release
cmake --install "${BUILD_DIR}"

export DL_3RD_DIR="${REPO_DIR}/3rdparty/target/Linux_x86_64"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib":"${DL_3RD_DIR}/onnxruntime/lib"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd "${INSTALL_DIR}"
echo "Running benchmarks..."
./benchmarks/ai_core_benchmarks

echo "Build and test process completed successfully."
