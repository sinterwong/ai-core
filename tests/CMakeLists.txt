cmake_minimum_required(VERSION 3.18)

project(Tests)

include("${AI_CORE_CMAKE_DIR}/dependencies/DevTools.cmake")
include("${AI_CORE_CMAKE_DIR}/dependencies/OpenCV.cmake")
include("${AI_CORE_CMAKE_DIR}/dependencies/Platform.cmake")
ai_core_load_googletest()
ai_core_load_opencv()
ai_core_load_platform()

# ============================================================================
# Source Files
# ============================================================================
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cc")

# ============================================================================
# Dependencies Configuration
# ============================================================================
# Base dependencies
set(DEPENDENCY_INCLUDES
    ${PROJECT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${AI_CORE_SOURCE_DIR}/plugins/common/include
    ${AI_CORE_SOURCE_DIR}/plugins/common/opencv
    ${AI_CORE_SOURCE_DIR}/plugins/preproc/opencv
    ${AI_CORE_SOURCE_DIR}/plugins/postproc/opencv
    ${AI_CORE_SOURCE_DIR}/plugins/infer/onnxruntime
    ${AI_CORE_SOURCE_DIR}/plugins/infer/ncnn
    ${AI_CORE_SOURCE_DIR}/plugins/infer/tensorrt
)

set(DEPENDENCY_LIBS
    ai_core::ai_core
    ai_core::opencv_imgcodecs
    ai_core::platform
    GTest::gtest
    GTest::gtest_main
)

if(AI_CORE_BUILD_CONFIG)
    list(APPEND DEPENDENCY_LIBS ai_core::config)
endif()

# Engine-specific dependencies
if(AI_CORE_BUILD_PLUGIN_ONNXRUNTIME)
    list(APPEND DEPENDENCY_LIBS
        onnxruntime::onnxruntime
        ai_core::plugin_onnxruntime)
    list(APPEND TEST_PLUGIN_DEFINITIONS
        AI_CORE_TEST_ORT_PLUGIN="$<TARGET_FILE:ai_core_plugin_onnxruntime>")
endif()

if(AI_CORE_BUILD_PLUGIN_NCNN)
    list(APPEND DEPENDENCY_LIBS
        ai_core::ncnn
        ai_core::plugin_ncnn)
    list(APPEND TEST_PLUGIN_DEFINITIONS
        AI_CORE_TEST_NCNN_PLUGIN="$<TARGET_FILE:ai_core_plugin_ncnn>")
endif()

if(AI_CORE_BUILD_PLUGIN_TENSORRT)
    list(APPEND DEPENDENCY_LIBS
        TensorRT::nvinfer
        CUDA::cudart_static
        ai_core::plugin_tensorrt)
    list(APPEND TEST_PLUGIN_DEFINITIONS
        AI_CORE_TEST_TRT_PLUGIN="$<TARGET_FILE:ai_core_plugin_tensorrt>")
endif()

# ============================================================================
# Test Executable
# ============================================================================
# Set output directory
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/tests)

# Create test executable
add_executable(ai_core_tests ${TEST_SOURCES})

add_library(ai_core_test_bad_version_plugin SHARED
    plugins/bad_version_plugin.cpp)
target_compile_features(ai_core_test_bad_version_plugin PRIVATE cxx_std_20)
target_link_libraries(ai_core_test_bad_version_plugin PRIVATE ai_core::ai_core)
set_target_properties(ai_core_test_bad_version_plugin PROPERTIES
    OUTPUT_NAME ai_core_plugin_test_bad_version
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib/test-plugins
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/test-plugins)

# Set C++ standard
target_compile_features(ai_core_tests PRIVATE cxx_std_20)

# Configure include directories
target_include_directories(ai_core_tests PRIVATE ${DEPENDENCY_INCLUDES})

# Link dependencies
target_link_libraries(ai_core_tests PRIVATE ${DEPENDENCY_LIBS})
target_compile_definitions(ai_core_tests PRIVATE ${TEST_PLUGIN_DEFINITIONS})

if(TARGET ai_core_plugin_preproc_opencv AND
   TARGET ai_core_plugin_postproc_opencv)
    target_link_libraries(ai_core_tests PRIVATE
        ai_core::plugin_preproc_opencv
        ai_core::plugin_postproc_opencv)
    add_dependencies(ai_core_tests
        ai_core_plugin_preproc_opencv
        ai_core_plugin_postproc_opencv
        ai_core_test_bad_version_plugin)
    target_compile_definitions(ai_core_tests PRIVATE
        AI_CORE_TEST_PREPROC_PLUGIN="$<TARGET_FILE:ai_core_plugin_preproc_opencv>"
        AI_CORE_TEST_POSTPROC_PLUGIN="$<TARGET_FILE:ai_core_plugin_postproc_opencv>"
        AI_CORE_TEST_BAD_VERSION_PLUGIN="$<TARGET_FILE:ai_core_test_bad_version_plugin>"
    )
endif()

# ============================================================================
# Compile Definitions
# ============================================================================
if(AI_CORE_BUILD_PLUGIN_ONNXRUNTIME)
    target_compile_definitions(ai_core_tests PRIVATE WITH_ORT)
endif()

if(AI_CORE_BUILD_PLUGIN_NCNN)
    target_compile_definitions(ai_core_tests PRIVATE WITH_NCNN)
endif()

if(AI_CORE_BUILD_PLUGIN_TENSORRT)
    target_compile_definitions(ai_core_tests PRIVATE WITH_TRT)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS ai_core_tests DESTINATION tests COMPONENT tests)
