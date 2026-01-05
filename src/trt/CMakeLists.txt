cmake_minimum_required(VERSION 3.18)

project(TrtEngineModule VERSION ${AI_CORE_VERSION} LANGUAGES CXX CUDA)

# Enable position independent code for object library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Source Files
# ============================================================================
# C++ sources
set(TRT_MODULE_CXX_SOURCES
    dnn_infer.cpp
    trt_utils.cpp
)

# CUDA sources
set(TRT_MODULE_CUDA_SOURCES
    cuda_utils.cu
    device_buffer_impl.cu
    gpu_generic_cuda_preprocessor.cu
)

# Headers
set(TRT_MODULE_HEADERS
    cuda_utils.hpp
    dnn_infer.hpp
    gpu_generic_cuda_preprocessor.hpp
    trt_utils.hpp
)

# ============================================================================
# Dependencies
# ============================================================================
set(TRT_MODULE_PRIVATE_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(TRT_MODULE_LINK_LIBS
    TensorRT::nvinfer
    ${CUDA_LIBRARIES}
    encrypt::encrypt
    ${OpenCV_LIBS}
)

# ============================================================================
# TensorRT Engine Module (Object Library with CUDA)
# ============================================================================
message(STATUS "Building TensorRT engine module as OBJECT library with CUDA support")

add_library(ai_trt_module OBJECT
    ${TRT_MODULE_CXX_SOURCES}
    ${TRT_MODULE_CUDA_SOURCES}
    ${TRT_MODULE_HEADERS}
)

# Set CUDA properties
set_target_properties(ai_trt_module PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Set C++ standard
target_compile_features(ai_trt_module PUBLIC cxx_std_20)

# Compiler options for CUDA and C++
target_compile_options(ai_trt_module PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -pthread,-lmpi_cxx,-lmpi
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++20
    >
)

# Configure include directories
target_include_directories(ai_trt_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api>
    PRIVATE
        ${TRT_MODULE_PRIVATE_INCLUDES}
)

# Link dependencies
target_link_libraries(ai_trt_module PRIVATE
    encrypt::encrypt
)

# ============================================================================
# Export Variables to Parent Scope
# ============================================================================
# Export link libraries
set(TRT_MODULE_LIBS ${TRT_MODULE_LINK_LIBS} PARENT_SCOPE)

# Export properties required by the consumer (ai_core library)
# The final target must enable device symbol resolution for CUDA linking
set(AI_TRT_CONSUMER_PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CACHE STRING "Properties required by the consumer of ai_trt_module OBJECT library"
)
set(AI_TRT_CONSUMER_PROPERTIES ${AI_TRT_CONSUMER_PROPERTIES} PARENT_SCOPE)

# Export compile options required by the consumer
set(AI_TRT_CONSUMER_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -pthread,-lmpi_cxx,-lmpi
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++20
    >
    CACHE STRING "Compile options required by the consumer of ai_trt_module OBJECT library"
)
set(AI_TRT_CONSUMER_COMPILE_OPTIONS ${AI_TRT_CONSUMER_COMPILE_OPTIONS} PARENT_SCOPE)
