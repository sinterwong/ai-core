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

set(AI_CORE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(AI_CORE_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Use standard installation directories
include(GNUInstallDirs)

# ============================================================================
# Build Options
# ============================================================================
option(AI_CORE_BUILD_CONFIG "Build the JSON config module (ai_core::config)" OFF)
option(AI_CORE_BUILD_PLUGIN_PREPROC_OPENCV
    "Build the OpenCV preprocessing plugin" OFF)
option(AI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV
    "Build the OpenCV postprocessing plugin" OFF)
option(AI_CORE_BUILD_PLUGIN_ONNXRUNTIME
    "Build the ONNX Runtime inference plugin" OFF)
option(AI_CORE_BUILD_PLUGIN_NCNN "Build the NCNN inference plugin" OFF)
option(AI_CORE_BUILD_PLUGIN_TENSORRT
    "Build the TensorRT inference plugin" OFF)
option(AI_CORE_ENABLE_MODEL_DECRYPTION
    "Enable encrypted-model support in inference plugins" OFF)
option(AI_CORE_BUILD_TESTS "Build tests" OFF)
option(AI_CORE_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(AI_CORE_BUILD_EXAMPLES "Build examples" OFF)

set(AI_CORE_BUILD_ANY_PLUGIN OFF)
if(AI_CORE_BUILD_PLUGIN_PREPROC_OPENCV OR
   AI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV OR
   AI_CORE_BUILD_PLUGIN_ONNXRUNTIME OR
   AI_CORE_BUILD_PLUGIN_NCNN OR
   AI_CORE_BUILD_PLUGIN_TENSORRT)
    set(AI_CORE_BUILD_ANY_PLUGIN ON)
endif()

# ============================================================================
# Build Configuration
# ============================================================================
message(INFO "================================")
message(STATUS "AICore version: ${AI_CORE_VERSION}")
message(INFO "================================")
message(STATUS "Build configuration:")
message(STATUS "  Config:                ${AI_CORE_BUILD_CONFIG}")
message(STATUS "  OpenCV preprocessor:   ${AI_CORE_BUILD_PLUGIN_PREPROC_OPENCV}")
message(STATUS "  OpenCV postprocessor:  ${AI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV}")
message(STATUS "  ONNX Runtime plugin:   ${AI_CORE_BUILD_PLUGIN_ONNXRUNTIME}")
message(STATUS "  NCNN plugin:           ${AI_CORE_BUILD_PLUGIN_NCNN}")
message(STATUS "  TensorRT plugin:       ${AI_CORE_BUILD_PLUGIN_TENSORRT}")
message(STATUS "  Model decryption:      ${AI_CORE_ENABLE_MODEL_DECRYPTION}")
message(STATUS "  Tests:                 ${AI_CORE_BUILD_TESTS}")
message(STATUS "  Benchmarks:            ${AI_CORE_BUILD_BENCHMARKS}")
message(STATUS "  Examples:              ${AI_CORE_BUILD_EXAMPLES}")
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

include(cmake/dependencies/Common.cmake)

message(STATUS "")
message(STATUS "Platform information:")
message(STATUS "  Target OS:          ${TARGET_OS}")
message(STATUS "  Target architecture: ${TARGET_ARCH}")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")

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

# Main library
add_subdirectory(src)

if(AI_CORE_BUILD_ANY_PLUGIN)
    add_subdirectory(plugins)
endif()

# Optional components
if(AI_CORE_BUILD_TESTS)
    message(STATUS "Adding tests to build")
    add_subdirectory(tests)
endif()

if(AI_CORE_BUILD_BENCHMARKS)
    message(STATUS "Adding benchmarks to build")
    add_subdirectory(benchmarks)
endif()

if(AI_CORE_BUILD_EXAMPLES)
    message(STATUS "Adding examples to build")
    add_subdirectory(examples)
endif()

# Assets (test/benchmark resources)
if(AI_CORE_BUILD_TESTS OR AI_CORE_BUILD_BENCHMARKS)
    message(STATUS "Adding asset resources to build")
    add_subdirectory(assets)
endif()
