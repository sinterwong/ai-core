cmake_minimum_required(VERSION 3.18)

# Enable CUDA if TensorRT engine is enabled
if(WITH_TRT_ENGINE)
    project(AICoreSrc VERSION ${AI_CORE_VERSION} LANGUAGES CXX CUDA)
else()
    project(AICoreSrc VERSION ${AI_CORE_VERSION} LANGUAGES CXX)
endif()

# Enable position independent code for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Load Dependencies
# ============================================================================
load_opencv()

# Android-specific configuration
if(TARGET_OS STREQUAL "Android")
    load_android_env()
    list(APPEND ANDROID_JIN_LIBS mediandk)
endif()

# Base dependencies
set(DEPENDENCY_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
)

set(DEPENDENCY_LIBS
    encrypt::encrypt
    ${OpenCV_LIBS}
    ${ANDROID_JIN_LIBS}
)

# Public API headers
# Recursively find all headers in api/ai_core/
file(GLOB_RECURSE AI_CORE_PUBLIC_HEADERS CONFIGURE_DEPENDS 
    "api/ai_core/*.hpp"
)

# Private/Internal Headers and Sources
# A. Root directory files (Non-recursive to strictly avoid sub folders)
file(GLOB AI_CORE_ROOT_SRCS CONFIGURE_DEPENDS "*.cpp")
file(GLOB AI_CORE_ROOT_HDRS CONFIGURE_DEPENDS "*.hpp")

# B. Implementation Subdirectories (Recursive to allow sub-folders)
# Includes: preproc, postproc, registrar
file(GLOB_RECURSE AI_CORE_SUBMOD_SRCS CONFIGURE_DEPENDS 
    "preproc/*.cpp"
    "postproc/*.cpp"
    "registrar/*.cpp"
)

file(GLOB_RECURSE AI_CORE_SUBMOD_HDRS CONFIGURE_DEPENDS 
    "preproc/*.hpp"
    "postproc/*.hpp"
    "registrar/*.hpp"
)

# Combine lists
set(AI_CORE_SOURCES
    ${AI_CORE_ROOT_SRCS}
    ${AI_CORE_SUBMOD_SRCS}
)

set(AI_CORE_PRIVATE_HEADERS
    ${AI_CORE_ROOT_HDRS}
    ${AI_CORE_SUBMOD_HDRS}
)

# ONNX Runtime engine
if(WITH_ORT_ENGINE)
    message(STATUS "Configuring ONNX Runtime engine module")
    load_onnxruntime()
    add_subdirectory(ort)
    list(APPEND DEPENDENCY_LIBS onnxruntime::onnxruntime)
endif()

# NCNN engine
if(WITH_NCNN_ENGINE)
    message(STATUS "Configuring NCNN engine module")
    load_openmp()
    load_ncnn()
    add_subdirectory(ncnn)

    list(APPEND DEPENDENCY_INCLUDES
        ${NCNN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn
    )
    list(APPEND DEPENDENCY_LIBS ${NCNN_MODULE_LIBS})
endif()

# TensorRT engine
if(WITH_TRT_ENGINE)
    message(STATUS "Configuring TensorRT engine module")
    load_cuda()
    load_tensorrt()

    # CUDA architectures: overridable via AI_CORE_CUDA_ARCHITECTURES,
    # defaults to the GPUs present on the build machine.
    if(AI_CORE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${AI_CORE_CUDA_ARCHITECTURES})
    else()
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    add_subdirectory(trt)

    list(APPEND DEPENDENCY_INCLUDES
        ${TensorRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trt
    )
    list(APPEND DEPENDENCY_LIBS ${TRT_MODULE_LIBS})
endif()

add_library(ai_core SHARED
    ${AI_CORE_SOURCES}
    ${AI_CORE_PRIVATE_HEADERS}

    # Conditionally include engine module object files
    $<$<BOOL:${WITH_ORT_ENGINE}>:$<TARGET_OBJECTS:ai_ort_module>>
    $<$<BOOL:${WITH_NCNN_ENGINE}>:$<TARGET_OBJECTS:ai_ncnn_module>>
    $<$<BOOL:${WITH_TRT_ENGINE}>:$<TARGET_OBJECTS:ai_trt_module>>
)

# Create namespaced alias for modern CMake usage
add_library(ai_core::ai_core ALIAS ai_core)

# Set C++ standard using target property (modern CMake style)
target_compile_features(ai_core PUBLIC cxx_std_20)

# Apply TensorRT-specific properties if enabled
if(WITH_TRT_ENGINE)
    message(STATUS "Applying TensorRT consumer properties to ai_core")
    set_target_properties(ai_core PROPERTIES ${AI_TRT_CONSUMER_PROPERTIES})
    target_compile_options(ai_core PRIVATE ${AI_TRT_CONSUMER_COMPILE_OPTIONS})
endif()

# Configure include directories
target_include_directories(ai_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${DEPENDENCY_INCLUDES}
)

# Link dependencies
# opencv_core is public: the API headers expose cv::Rect / cv::Mat types
target_link_libraries(ai_core
    PUBLIC opencv_core
    PRIVATE ${DEPENDENCY_LIBS}
)

# ============================================================================
# Compile Definitions
# ============================================================================
# Define macros based on enabled engines
if(WITH_ORT_ENGINE)
    target_compile_definitions(ai_core PRIVATE WITH_ORT)
endif()

if(WITH_NCNN_ENGINE)
    target_compile_definitions(ai_core PRIVATE WITH_NCNN)
endif()

if(WITH_TRT_ENGINE)
    target_compile_definitions(ai_core PRIVATE WITH_TRT)
endif()

# ============================================================================
# Installation
# ============================================================================
# Install library targets
install(
    TARGETS ai_core
    EXPORT ai_core_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)

# Install public headers
install(
    FILES ${AI_CORE_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ai_core
    COMPONENT dev
)

# ============================================================================
# CMake Package Configuration
# ============================================================================
include(CMakePackageConfigHelpers)

set_target_properties(ai_core PROPERTIES
    VERSION ${AI_CORE_VERSION}
    SOVERSION ${AI_CORE_VERSION_MAJOR}
)

# Generate version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Export targets for build tree
export(
    EXPORT ai_core_export_set
    FILE "${CMAKE_CURRENT_BINARY_DIR}/ai_core-targets.cmake"
    NAMESPACE ai_core::
)

# Configure package config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai_core-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config.cmake"
    @ONLY
)

# Install package configuration files
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_DATADIR}
    COMPONENT dev
)

# Install targets for install tree
install(
    EXPORT ai_core_export_set
    FILE ai_core-targets.cmake
    NAMESPACE ai_core::
    DESTINATION ${CMAKE_INSTALL_DATADIR}
    COMPONENT dev
)