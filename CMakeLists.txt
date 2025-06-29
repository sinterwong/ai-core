CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Project setting
INCLUDE(cmake/messagecolor.cmake)
INCLUDE(cmake/utils.cmake)
AI_CORE_EXTRACT_VERSION()

PROJECT(AICore VERSION ${AI_CORE_VERSION} LANGUAGES CXX)

INCLUDE(GNUInstallDirs)

SET(CMAKE_CXX_STANDARD 20)

OPTION(BUILD_AI_CORE_TESTS "Build with tests" OFF)

OPTION(WITH_ORT_ENGINE "Build with onnxruntime" ON)
OPTION(WITH_NCNN_ENGINE "Build with ncnn" OFF)

MESSAGE(INFO "--------------------------------")
MESSAGE(STATUS "Build AICore: ${AI_CORE_VERSION}")
MESSAGE(STATUS "Build with tests: ${BUILD_AI_CORE_TESTS}")
MESSAGE(STATUS "Build with onnxruntime: ${WITH_ORT_ENGINE}")
MESSAGE(STATUS "Build with ncnn: ${WITH_NCNN_ENGINE}")
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
MESSAGE(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

IF(MSVC)
    ADD_COMPILE_OPTIONS(/utf-8)
ENDIF()

IF(CMAKE_TOOLCHAIN_FILE)
    MESSAGE(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
ENDIF()

# Set default build to debug
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose Release, Debug, RelWithDebInfo or MinSizeRel" FORCE)
ENDIF()

IF(NOT TARGET_OS)
    SET(TARGET_OS ${CMAKE_SYSTEM_NAME})
ENDIF()

IF(NOT TARGET_ARCH)
    SET(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()


MESSAGE(STATUS "Your target os : ${TARGET_OS}")
MESSAGE(STATUS "Your target arch : ${TARGET_ARCH}")

MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Set compilation options
MESSAGE(INFO "--------------------------------")
INCLUDE(cmake/load_3rdparty.cmake)
MESSAGE(INFO "Loading 3rdparty libraries")

SET(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${TARGET_ARCH})
SET(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

MESSAGE("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
MESSAGE("-- output path:  ${PROJECT_OUTPUT_DIR}")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# Build 3rdparty
MESSAGE(INFO "3rdparty libraries has loaded.")
MESSAGE(INFO "--------------------------------")
ADD_SUBDIRECTORY(3rdparty)

# Build application
MESSAGE(INFO "--------------------------------")
ADD_SUBDIRECTORY(src)

IF(BUILD_AI_CORE_TESTS)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(tests)
ENDIF()
