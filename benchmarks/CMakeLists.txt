cmake_minimum_required(VERSION 3.18)

project(Benchmarks)

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
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/plugins/common/include
    ${CMAKE_SOURCE_DIR}/plugins/common/opencv
    ${CMAKE_SOURCE_DIR}/plugins/preproc/opencv
    ${CMAKE_SOURCE_DIR}/plugins/postproc/opencv
    ${CMAKE_SOURCE_DIR}/plugins/infer/onnxruntime
    ${CMAKE_SOURCE_DIR}/plugins/infer/ncnn
    ${CMAKE_SOURCE_DIR}/plugins/infer/tensorrt
)

set(DEPENDENCY_LIBS
    ai_core::ai_core
    encrypt::encrypt
    benchmark::benchmark
    ${OpenCV_LIBS}
    opencv_imgcodecs
    ${ANDROID_JIN_LIBS}
)

if(TARGET ai_core_plugin_preproc_opencv)
    list(APPEND DEPENDENCY_LIBS ai_core::plugin_preproc_opencv)
endif()

if(TARGET ai_core_plugin_postproc_opencv)
    list(APPEND DEPENDENCY_LIBS ai_core::plugin_postproc_opencv)
endif()

# Engine-specific dependencies
if(WITH_ORT_ENGINE)
    load_onnxruntime()
    list(APPEND DEPENDENCY_INCLUDES
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime
    )
    list(APPEND DEPENDENCY_LIBS
        onnxruntime::onnxruntime
        ai_core::plugin_onnxruntime)
endif()

if(WITH_NCNN_ENGINE)
    load_openmp()
    load_ncnn()
    list(APPEND DEPENDENCY_INCLUDES ${NCNN_INCLUDE_DIR})
    list(APPEND DEPENDENCY_LIBS
        ${NCNN_LIBS}
        OpenMP::OpenMP_CXX
        ai_core::plugin_ncnn)
endif()

if(WITH_TRT_ENGINE)
    load_cuda()
    load_tensorrt()
    list(APPEND DEPENDENCY_INCLUDES
        ${TensorRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trt
    )
    list(APPEND DEPENDENCY_LIBS
        TensorRT::nvinfer
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
if(WITH_ORT_ENGINE)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        WITH_ORT
        AI_CORE_BENCH_ORT_PLUGIN="$<TARGET_FILE:ai_core_plugin_onnxruntime>")
endif()

if(WITH_NCNN_ENGINE)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        WITH_NCNN
        AI_CORE_BENCH_NCNN_PLUGIN="$<TARGET_FILE:ai_core_plugin_ncnn>")
endif()

if(WITH_TRT_ENGINE)
    target_compile_definitions(ai_core_benchmarks PRIVATE
        WITH_TRT
        AI_CORE_BENCH_TRT_PLUGIN="$<TARGET_FILE:ai_core_plugin_tensorrt>")
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS ai_core_benchmarks DESTINATION benchmarks)
