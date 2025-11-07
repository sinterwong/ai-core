cmake_minimum_required(VERSION 3.18)

project(ncnnEngineModule)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NCNN_MODULE_PRIVATE_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${NCNN_INCLUDE_DIR}
)

set(NCNN_MODULE_LINK_LIBS
    encrypt::encrypt
    logger::logger
    ${OpenMP_CXX_FLAGS}
    ${NCNN_LIBS}
    ${OpenCV_LIBS}
)

set(NCNN_MODULE_LIBS ${NCNN_MODULE_LINK_LIBS} PARENT_SCOPE)

file(GLOB_RECURSE CURRENT_DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

message(STATUS "Inference engine type is NCNN, preparing object files.")

add_library(ai_ncnn_module OBJECT
    ${CURRENT_DIR_SRCS}
)

target_include_directories(ai_ncnn_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api>
    PRIVATE
        ${NCNN_MODULE_PRIVATE_INCLUDES}
)

target_link_libraries(ai_ncnn_module PRIVATE
    encrypt::encrypt
    logger::logger
)
