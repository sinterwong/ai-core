cmake_minimum_required(VERSION 3.18)

project(OrtEngineModule)

# Enable position independent code for object library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Source Files
# ============================================================================
set(ORT_MODULE_SOURCES
    ort_infer.cpp
)

set(ORT_MODULE_HEADERS
    ort_infer.hpp
)

# ============================================================================
# Dependencies
# ============================================================================
set(ORT_MODULE_PRIVATE_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
)

set(ORT_MODULE_LINK_LIBS
    onnxruntime::onnxruntime
    ${OpenCV_LIBS}
    encrypt::encrypt
)

# Export link libraries to parent scope
set(ORT_MODULE_LIBS ${ORT_MODULE_LINK_LIBS} PARENT_SCOPE)

# ============================================================================
# ONNX Runtime Engine Module (Object Library)
# ============================================================================
message(STATUS "Building ONNX Runtime engine module as OBJECT library")

add_library(ai_ort_module OBJECT
    ${ORT_MODULE_SOURCES}
    ${ORT_MODULE_HEADERS}
)

# Set C++ standard
target_compile_features(ai_ort_module PUBLIC cxx_std_20)

# Configure include directories
target_include_directories(ai_ort_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api>
    PRIVATE
        ${ORT_MODULE_PRIVATE_INCLUDES}
)

# Link dependencies
target_link_libraries(ai_ort_module PRIVATE
    onnxruntime::onnxruntime
    encrypt::encrypt
)
