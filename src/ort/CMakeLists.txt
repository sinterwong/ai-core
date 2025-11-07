cmake_minimum_required(VERSION 3.18)

project(OrtEngineModule)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ORT_MODULE_PRIVATE_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
)

set(ORT_MODULE_LINK_LIBS
    onnxruntime::onnxruntime
    ${OpenCV_LIBS}
    encrypt::encrypt
    logger::logger
)

set(ORT_MODULE_LIBS ${ORT_MODULE_LINK_LIBS} PARENT_SCOPE)

file(GLOB_RECURSE CURRENT_DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

message(STATUS "Inference engine type is ONNXRuntime, preparing object files.")

add_library(ai_ort_module OBJECT
    ${CURRENT_DIR_SRCS}
)

target_include_directories(ai_ort_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api>
    PRIVATE
        ${ORT_MODULE_PRIVATE_INCLUDES}
)

target_link_libraries(ai_ort_module PRIVATE
    onnxruntime::onnxruntime
    encrypt::encrypt
    logger::logger
)
