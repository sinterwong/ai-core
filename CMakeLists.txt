cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project setting
include(cmake/messagecolor.cmake)
include(cmake/utils.cmake)
ai_core_extract_version()

project(AICore VERSION ${AI_CORE_VERSION} LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_AI_CORE_TESTS "Build with tests" OFF)
option(BUILD_AI_CORE_BENCHMARKS "Build with benchmarks" OFF)
option(BUILD_AI_CORE_EXAMPLES "Build with examples" OFF)

option(WITH_ORT_ENGINE "Build with onnxruntime" ON)
option(WITH_NCNN_ENGINE "Build with ncnn" OFF)
option(WITH_TRT_ENGINE "Build with tensorrt" OFF)

message(INFO "--------------------------------")
message(STATUS "Build AICore: ${AI_CORE_VERSION}")
message(STATUS "Build with tests: ${BUILD_AI_CORE_TESTS}")
message(STATUS "Build with benchmarks: ${BUILD_AI_CORE_BENCHMARKS}")
message(STATUS "Build with examples: ${BUILD_AI_CORE_EXAMPLES}")
message(STATUS "Build with onnxruntime: ${WITH_ORT_ENGINE}")
message(STATUS "Build with ncnn: ${WITH_NCNN_ENGINE}")
message(STATUS "Build with tensorrt: ${WITH_TRT_ENGINE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

if(MSVC)
    add_compile_options(/utf-8)
endif()

if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Set default build to RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose Release, Debug, RelWithDebInfo or MinSizeRel" FORCE)
endif()

if(NOT TARGET_OS)
    set(TARGET_OS ${CMAKE_SYSTEM_NAME})
endif()

if(NOT TARGET_ARCH)
    set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

message(STATUS "Target OS: ${TARGET_OS}")
message(STATUS "Target architecture: ${TARGET_ARCH}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Load 3rdparty libraries
message(INFO "--------------------------------")
message(INFO "Loading 3rdparty libraries")
include(cmake/load_3rdparty.cmake)

set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${TARGET_ARCH})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message(STATUS "System architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Output path: ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# Build 3rdparty
message(INFO "3rdparty libraries has loaded.")
message(INFO "--------------------------------")
add_subdirectory(3rdparty)

# Build application
message(INFO "--------------------------------")
add_subdirectory(src)

if(BUILD_AI_CORE_TESTS)
    message(INFO "------------- Build tests --------------")
    add_subdirectory(tests)
endif()

if(BUILD_AI_CORE_BENCHMARKS)
    message(INFO "------------- Build benchmarks --------------")
    add_subdirectory(benchmarks)
endif()

if(BUILD_AI_CORE_EXAMPLES)
    message(INFO "------------- Build examples --------------")
    add_subdirectory(examples)
endif()

if(BUILD_AI_CORE_TESTS OR BUILD_AI_CORE_BENCHMARKS)
    message(INFO "--------- Install Resource --------")
    add_subdirectory(assets)
endif()
