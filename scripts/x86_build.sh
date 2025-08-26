#!/bin/bash

# This script is used to build the ai-core project for x86_64 architecture.

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

echo "Creating 3rdparty target directory..."
mkdir -p ${REPO_DIR}/3rdparty/target

echo "Downloading dependency archive..."
curl -L https://github.com/sinterwong/ai-core/releases/download/v1.0.0-alpha/dependency-Linux_x86_64.tgz -o dependency.tgz

echo "Extracting dependency archive..."
tar xvf dependency.tgz -C ${REPO_DIR}/3rdparty/target

echo "Removing dependency archive..."
rm dependency.tgz

echo "Updating apt packages and installing build tools..."
sudo apt-get update
sudo apt-get install -y gcc-13 g++-13
sudo apt-get install -y ninja-build

echo "Configuring CMake project..."
cmake -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_AI_CORE_TESTS=ON \
    -DBUILD_AI_CORE_BENCHMARKS=ON \
    -DWITH_ORT_ENGINE=ON \
    -DWITH_NCNN_ENGINE=ON \
    -DWITH_TRT_ENGINE=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -S ${REPO_DIR}

echo "Building project..."
cmake --build "${BUILD_DIR}" --config Release

echo "Installing project..."
cmake --install "${BUILD_DIR}"

echo "Setting up environment variables for tests..."
export BUILD_3RD_DIR="${BUILD_DIR}/3rdparty"
export DL_3RD_DIR="${REPO_DIR}/3rdparty/target/Linux_x86_64"

export LD_LIBRARY_PATH="${INSTALL_DIR}/lib":"${BUILD_3RD_DIR}/encryption-tool/x86_64/lib":"${BUILD_3RD_DIR}/logger/x86_64/lib":"${DL_3RD_DIR}/onnxruntime/lib":"${DL_3RD_DIR}/opencv/lib"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo $LD_LIBRARY_PATH

cd "${INSTALL_DIR}"
echo "Running tests..."
./tests/ai_core_tests --gtest_filter=*.*

echo "Running benchmarks..."
./benchmarks/ai_core_benchmarks

echo "Build and test process completed successfully."
