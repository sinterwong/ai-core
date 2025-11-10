cmake_minimum_required(VERSION 3.18)

project(ncnnEngineModule)

# Enable position independent code for object library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Source Files
# ============================================================================
set(NCNN_MODULE_SOURCES
    dnn_infer.cpp
    ncnn_generic_preprocessor.cpp
)

set(NCNN_MODULE_HEADERS
    dnn_infer.hpp
    ncnn_generic_preprocessor.hpp
)

# ============================================================================
# Dependencies
# ============================================================================
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

# Export link libraries to parent scope
set(NCNN_MODULE_LIBS ${NCNN_MODULE_LINK_LIBS} PARENT_SCOPE)

# ============================================================================
# NCNN Engine Module (Object Library)
# ============================================================================
message(STATUS "Building NCNN engine module as OBJECT library")

add_library(ai_ncnn_module OBJECT
    ${NCNN_MODULE_SOURCES}
    ${NCNN_MODULE_HEADERS}
)

# Set C++ standard
target_compile_features(ai_ncnn_module PUBLIC cxx_std_20)

# Configure include directories
target_include_directories(ai_ncnn_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../api>
    PRIVATE
        ${NCNN_MODULE_PRIVATE_INCLUDES}
)

# Link dependencies
target_link_libraries(ai_ncnn_module PRIVATE
    encrypt::encrypt
    logger::logger
)
