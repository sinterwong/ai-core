cmake_minimum_required(VERSION 3.18)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Load custom CMake modules
include(cmake/messagecolor.cmake)
include(cmake/utils.cmake)

# Extract version from source header
ai_core_extract_version()

# Declare project
project(AICore VERSION ${AI_CORE_VERSION} LANGUAGES CXX)

# Use standard installation directories
include(GNUInstallDirs)

# ============================================================================
# Build Options
# ============================================================================
option(BUILD_AI_CORE_TESTS "Build unit tests" OFF)
option(BUILD_AI_CORE_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_AI_CORE_EXAMPLES "Build examples" OFF)
option(BUILD_AI_CORE_CONFIG "Build the JSON config module (ai_core::config)" ON)

# Inference engine options
option(WITH_ORT_ENGINE "Build with ONNX Runtime engine" ON)
option(WITH_NCNN_ENGINE "Build with NCNN engine" OFF)
option(WITH_TRT_ENGINE "Build with TensorRT engine" OFF)

# ============================================================================
# Build Configuration
# ============================================================================
message(INFO "================================")
message(STATUS "AICore version: ${AI_CORE_VERSION}")
message(INFO "================================")
message(STATUS "Build configuration:")
message(STATUS "  Tests:       ${BUILD_AI_CORE_TESTS}")
message(STATUS "  Benchmarks:  ${BUILD_AI_CORE_BENCHMARKS}")
message(STATUS "  Examples:    ${BUILD_AI_CORE_EXAMPLES}")
message(STATUS "")
message(STATUS "Inference engines:")
message(STATUS "  ONNX Runtime: ${WITH_ORT_ENGINE}")
message(STATUS "  NCNN:         ${WITH_NCNN_ENGINE}")
message(STATUS "  TensorRT:     ${WITH_TRT_ENGINE}")
message(STATUS "")
message(STATUS "Installation prefix: ${CMAKE_INSTALL_PREFIX}")

# ============================================================================
# Compiler Configuration
# ============================================================================
# MSVC: Enable UTF-8 source file encoding
if(MSVC)
    add_compile_options(/utf-8)
endif()

# Display toolchain file if specified
if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose Release, Debug, RelWithDebInfo, or MinSizeRel" FORCE)
endif()

# Determine target platform and architecture
if(NOT TARGET_OS)
    set(TARGET_OS ${CMAKE_SYSTEM_NAME})
endif()

if(NOT TARGET_ARCH)
    set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

message(STATUS "")
message(STATUS "Platform information:")
message(STATUS "  Target OS:          ${TARGET_OS}")
message(STATUS "  Target architecture: ${TARGET_ARCH}")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")

# ============================================================================
# Third-Party Dependencies
# ============================================================================
message(INFO "================================")
message(INFO "Loading third-party dependencies")
message(INFO "================================")
include(cmake/load_3rdparty.cmake)

# ============================================================================
# Output Directories
# ============================================================================
# Architecture-specific output directory
set(PROJECT_OUTPUT_DIR ${PROJECT_BINARY_DIR}/${TARGET_ARCH})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

# Create output directories
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

message(STATUS "")
message(STATUS "Output directories:")
message(STATUS "  Base:     ${PROJECT_OUTPUT_DIR}")
message(STATUS "  Binaries: ${PROJECT_OUTPUT_DIR}/bin")
message(STATUS "  Libraries: ${PROJECT_OUTPUT_DIR}/lib")
message(STATUS "  Headers:  ${PROJECT_INCLUDE_DIR}")

# Set standard output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# ============================================================================
# Build Subdirectories
# ============================================================================
message(INFO "================================")
message(INFO "Building project components")
message(INFO "================================")

# Third-party libraries
add_subdirectory(3rdparty)

# Main library
add_subdirectory(src)

# Optional components
if(BUILD_AI_CORE_TESTS)
    message(STATUS "Adding tests to build")
    add_subdirectory(tests)
endif()

if(BUILD_AI_CORE_BENCHMARKS)
    message(STATUS "Adding benchmarks to build")
    add_subdirectory(benchmarks)
endif()

if(BUILD_AI_CORE_EXAMPLES)
    message(STATUS "Adding examples to build")
    add_subdirectory(examples)
endif()

# Assets (test/benchmark resources)
if(BUILD_AI_CORE_TESTS OR BUILD_AI_CORE_BENCHMARKS)
    message(STATUS "Adding asset resources to build")
    add_subdirectory(assets)
endif()
