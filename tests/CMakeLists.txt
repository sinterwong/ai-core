cmake_minimum_required(VERSION 3.18)

project(Tests)

# ============================================================================
# Load Dependencies
# ============================================================================
load_opencv()

# Android-specific configuration
if(TARGET_OS STREQUAL "Android")
    load_android_env()
    list(APPEND ANDROID_JIN_LIBS mediandk)
endif()

# ============================================================================
# Source Files
# ============================================================================
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp" ".cc")

# ============================================================================
# Dependencies Configuration
# ============================================================================
# Base dependencies
set(DEPENDENCY_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ort
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ncnn
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/trt
)

set(DEPENDENCY_LIBS
    ai_core::ai_core
    encrypt::encrypt
    ${OpenCV_LIBS}
    GTest::gtest
    GTest::gtest_main
    ${ANDROID_JIN_LIBS}
)

if(BUILD_AI_CORE_CONFIG)
    list(APPEND DEPENDENCY_LIBS ai_core::config)
endif()

# Engine-specific dependencies
if(WITH_ORT_ENGINE)
    load_onnxruntime()
    list(APPEND DEPENDENCY_LIBS onnxruntime::onnxruntime)
endif()

if(WITH_NCNN_ENGINE)
    load_openmp()
    load_ncnn()
    list(APPEND DEPENDENCY_INCLUDES ${NCNN_INCLUDE_DIR})
    list(APPEND DEPENDENCY_LIBS ${NCNN_LIBS} ${OpenMP_CXX_FLAGS})
endif()

if(WITH_TRT_ENGINE)
    load_cuda()
    load_tensorrt()
    list(APPEND DEPENDENCY_INCLUDES
        ${TensorRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trt
    )
    list(APPEND DEPENDENCY_LIBS TensorRT::nvinfer)
endif()

# ============================================================================
# Test Executable
# ============================================================================
# Set output directory
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/tests)

# Create test executable
add_executable(ai_core_tests ${TEST_SOURCES})

# Set C++ standard
target_compile_features(ai_core_tests PRIVATE cxx_std_20)

# Configure include directories
target_include_directories(ai_core_tests PRIVATE ${DEPENDENCY_INCLUDES})

# Link dependencies
target_link_libraries(ai_core_tests PRIVATE ${DEPENDENCY_LIBS})

# ============================================================================
# Compile Definitions
# ============================================================================
if(WITH_ORT_ENGINE)
    target_compile_definitions(ai_core_tests PRIVATE WITH_ORT)
endif()

if(WITH_NCNN_ENGINE)
    target_compile_definitions(ai_core_tests PRIVATE WITH_NCNN)
endif()

if(WITH_TRT_ENGINE)
    target_compile_definitions(ai_core_tests PRIVATE WITH_TRT)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS ai_core_tests DESTINATION tests)
