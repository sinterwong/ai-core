CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(TrtEngineModule VERSION ${AI_CORE_VERSION} LANGUAGES CXX CUDA)

SET(CMAKE_POSITION_INDEPENDENT_CODE ON) 

SET(TRT_MODULE_PRIVATE_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

SET(TRT_MODULE_LINK_LIBS
    TensorRT::nvinfer
    ${CUDA_LIBRARIES}
    encrypt::encrypt
    logger::logger
    ${OpenCV_LIBS}
)

FILE(GLOB_RECURSE CURRENT_DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

MESSAGE(STATUS "Inference engine type is TensorRT, preparing object files.")

ADD_LIBRARY(ai_trt_module OBJECT
    ${CURRENT_DIR_SRCS}
)

SET_TARGET_PROPERTIES(ai_trt_module PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

TARGET_COMPILE_OPTIONS(ai_trt_module PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -pthread,-lmpi_cxx,-lmpi
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++20
    >
)

TARGET_INCLUDE_DIRECTORIES(ai_trt_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api> 
    PRIVATE
        ${TRT_MODULE_PRIVATE_INCLUDES}
)

TARGET_LINK_LIBRARIES(ai_trt_module PRIVATE
    encrypt::encrypt
    common_utils::common_utils
    logger::logger
)

# 给父作用域传递必要的变量
SET(TRT_MODULE_LIBS ${TRT_MODULE_LINK_LIBS} PARENT_SCOPE)

# 上层的最终目标必须启用设备符号解析
SET(AI_TRT_CONSUMER_PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CACHE STRING "Properties required by the consumer of the ai_trt_module OBJECT library."
)

SET(AI_TRT_CONSUMER_PROPERTIES ${AI_TRT_CONSUMER_PROPERTIES} PARENT_SCOPE)

SET(AI_TRT_CONSUMER_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -pthread,-lmpi_cxx,-lmpi
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++20
    >
    CACHE STRING "Compile options required by the consumer of the ai_trt_module OBJECT library."
)

SET(AI_TRT_CONSUMER_COMPILE_OPTIONS ${AI_TRT_CONSUMER_COMPILE_OPTIONS} PARENT_SCOPE)