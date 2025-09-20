CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(OrtEngineModule)

SET(CMAKE_POSITION_INDEPENDENT_CODE ON) 

SET(ORT_MODULE_PRIVATE_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
)

SET(ORT_MODULE_LINK_LIBS
    onnxruntime::onnxruntime
    ${OpenCV_LIBS}
    encrypt::encrypt
    logger::logger
)

SET(ORT_MODULE_LIBS ${ORT_MODULE_LINK_LIBS} PARENT_SCOPE)

FILE(GLOB_RECURSE CURRENT_DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

MESSAGE(STATUS "Inference engine type is ONNXRuntime, preparing object files.")

ADD_LIBRARY(ai_ort_module OBJECT
    ${CURRENT_DIR_SRCS}
)

TARGET_INCLUDE_DIRECTORIES(ai_ort_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api> 
    PRIVATE
        ${ORT_MODULE_PRIVATE_INCLUDES}
)

TARGET_LINK_LIBRARIES(ai_ort_module PRIVATE
    onnxruntime::onnxruntime
    encrypt::encrypt
    logger::logger
)
