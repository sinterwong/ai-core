cmake_minimum_required(VERSION 3.18)

project(TrtEngineModule VERSION ${AI_CORE_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRT_MODULE_PRIVATE_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(TRT_MODULE_LINK_LIBS
    TensorRT::nvinfer
    ${CUDA_LIBRARIES}
    encrypt::encrypt
    logger::logger
    ${OpenCV_LIBS}
)

file(GLOB_RECURSE CURRENT_DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

message(STATUS "Inference engine type is TensorRT, preparing object files.")

add_library(ai_trt_module OBJECT
    ${CURRENT_DIR_SRCS}
)

set_target_properties(ai_trt_module PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(ai_trt_module PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -pthread,-lmpi_cxx,-lmpi
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++20
    >
)

target_include_directories(ai_trt_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api>
    PRIVATE
        ${TRT_MODULE_PRIVATE_INCLUDES}
)

target_link_libraries(ai_trt_module PRIVATE
    encrypt::encrypt
    logger::logger
)

# Export necessary variables to parent scope
set(TRT_MODULE_LIBS ${TRT_MODULE_LINK_LIBS} PARENT_SCOPE)

# The final target in parent scope must enable device symbol resolution
set(AI_TRT_CONSUMER_PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CACHE STRING "Properties required by the consumer of the ai_trt_module OBJECT library."
)

set(AI_TRT_CONSUMER_PROPERTIES ${AI_TRT_CONSUMER_PROPERTIES} PARENT_SCOPE)

set(AI_TRT_CONSUMER_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -pthread,-lmpi_cxx,-lmpi
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++20
    >
    CACHE STRING "Compile options required by the consumer of the ai_trt_module OBJECT library."
)

set(AI_TRT_CONSUMER_COMPILE_OPTIONS ${AI_TRT_CONSUMER_COMPILE_OPTIONS} PARENT_SCOPE)
