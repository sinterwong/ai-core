cmake_minimum_required(VERSION 3.18)

project(Benchmarks)

if(NOT TARGET ai_core_plugin_preproc_opencv OR
   NOT TARGET ai_core_plugin_postproc_opencv)
    message(FATAL_ERROR
        "Benchmarks require both OpenCV plugins. Enable "
        "AI_CORE_BUILD_PLUGIN_PREPROC_OPENCV and "
        "AI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV.")
endif()

include("${AI_CORE_CMAKE_DIR}/dependencies/DevTools.cmake")
include("${AI_CORE_CMAKE_DIR}/dependencies/OpenCV.cmake")
include("${AI_CORE_CMAKE_DIR}/dependencies/Platform.cmake")
ai_core_load_benchmark()
ai_core_load_opencv()
ai_core_load_platform()

# ============================================================================
# Source Files
# ============================================================================
set(BENCHMARK_SOURCES
    main.cc
    yolo_infer_bench.cpp
    yolo_postproc_bench.cpp
    yolo_preproc_bench.cpp
    trt_infer_bench.cpp
)

# ============================================================================
# Dependencies Configuration
# ============================================================================
# Base dependencies
set(DEPENDENCY_INCLUDES
    ${PROJECT_INCLUDE_DIR}
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
    benchmark::benchmark
    ai_core::opencv_imgcodecs
    ai_core::platform
)

if(TARGET ai_core_plugin_preproc_opencv)
    list(APPEND DEPENDENCY_LIBS ai_core::plugin_preproc_opencv)
endif()

if(TARGET ai_core_plugin_postproc_opencv)
    list(APPEND DEPENDENCY_LIBS ai_core::plugin_postproc_opencv)
endif()

# Engine-specific dependencies
if(AI_CORE_BUILD_PLUGIN_ONNXRUNTIME)
    list(APPEND DEPENDENCY_LIBS
        onnxruntime::onnxruntime
        ai_core::plugin_onnxruntime)
endif()

if(AI_CORE_BUILD_PLUGIN_NCNN)
    list(APPEND DEPENDENCY_LIBS
        ai_core::ncnn
        ai_core::plugin_ncnn)
endif()

if(AI_CORE_BUILD_PLUGIN_TENSORRT)
    list(APPEND DEPENDENCY_LIBS
        TensorRT::nvinfer
        CUDA::cudart_static
        ai_core::plugin_tensorrt)
endif()

# ============================================================================
# Benchmark Executable
# ============================================================================
# Set output directory
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/benchmarks)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/benchmarks)

# Create benchmark executable
add_executable(ai_core_benchmarks ${BENCHMARK_SOURCES})

# Set C++ standard
target_compile_features(ai_core_benchmarks PRIVATE cxx_std_20)

# Configure include directories
target_include_directories(ai_core_benchmarks PRIVATE ${DEPENDENCY_INCLUDES})

# Link dependencies
target_link_libraries(ai_core_benchmarks PRIVATE ${DEPENDENCY_LIBS})

if(TARGET ai_core_plugin_preproc_opencv)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        AI_CORE_BENCH_PREPROC_PLUGIN="$<TARGET_FILE:ai_core_plugin_preproc_opencv>")
endif()

if(TARGET ai_core_plugin_postproc_opencv)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        AI_CORE_BENCH_POSTPROC_PLUGIN="$<TARGET_FILE:ai_core_plugin_postproc_opencv>")
endif()

# ============================================================================
# Compile Definitions
# ============================================================================
if(AI_CORE_BUILD_PLUGIN_ONNXRUNTIME)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        WITH_ORT
        AI_CORE_BENCH_ORT_PLUGIN="$<TARGET_FILE:ai_core_plugin_onnxruntime>")
endif()

if(AI_CORE_BUILD_PLUGIN_NCNN)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        WITH_NCNN
        AI_CORE_BENCH_NCNN_PLUGIN="$<TARGET_FILE:ai_core_plugin_ncnn>")
endif()

if(AI_CORE_BUILD_PLUGIN_TENSORRT)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        WITH_TRT
        AI_CORE_BENCH_TRT_PLUGIN="$<TARGET_FILE:ai_core_plugin_tensorrt>")
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS ai_core_benchmarks DESTINATION benchmarks
    COMPONENT benchmarks)
