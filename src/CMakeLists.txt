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

# Private/Internal sources — explicitly listed (no GLOB) so adding a file is a
# deliberate, reviewable change and the build never silently picks up strays.
set(AI_CORE_SOURCES
    # facade + pimpl
    algo_infer.cpp
    algo_infer_impl.cpp
    algo_infer_engine.cpp
    algo_infer_engine_impl.cpp
    algo_manager.cpp
    algo_manager_impl.cpp
    algo_preproc.cpp
    algo_preproc_impl.cpp
    algo_postproc.cpp
    algo_postproc_impl.cpp
    # core value types + utilities
    typed_buffer.cpp
    vision_util.cpp
    logger.cpp
    infer_base.cpp
    device_buffer_impl_stub.cpp
    # preprocessing
    preproc/cpu_generic_preprocessor.cpp
    preproc/frame_with_mask_prep.cpp
    preproc/generic_frame_preproc_base.cpp
    # postprocessing
    postproc/yolo_det.cpp
    postproc/nano_det.cpp
    postproc/rtm_det.cpp
    postproc/softmax_cls.cpp
    postproc/fpr_cls.cpp
    postproc/semantic_seg.cpp
    postproc/unet_dual_out_seg.cpp
    postproc/ocr_reco.cpp
    postproc/raw_output.cpp
    # plugin registrars
    registrar/default_plugins.cpp
    registrar/preproc_registrar.cpp
    registrar/postproc_registrar.cpp
    registrar/infer_engine_registrar.cpp
)

# Private headers are only used for IDE indexing; globbing them is harmless.
file(GLOB AI_CORE_PRIVATE_HEADERS CONFIGURE_DEPENDS
    "*.hpp" "preproc/*.hpp" "postproc/*.hpp" "registrar/*.hpp")

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

# Install public headers (top-level only; subdir headers install with their
# module below to preserve the ai_core/<module>/ layout).
file(GLOB AI_CORE_PUBLIC_HEADERS_TOP CONFIGURE_DEPENDS "api/ai_core/*.hpp")
install(
    FILES ${AI_CORE_PUBLIC_HEADERS_TOP}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ai_core
    COMPONENT dev
)

# ============================================================================
# Optional module: ai_core::config (JSON pipeline loader)
# ============================================================================
if(BUILD_AI_CORE_CONFIG)
    message(STATUS "Configuring config module (ai_core::config)")
    add_library(ai_core_config SHARED config/algo_config.cpp)
    add_library(ai_core::config ALIAS ai_core_config)
    target_compile_features(ai_core_config PUBLIC cxx_std_20)
    target_include_directories(ai_core_config
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${PROJECT_INCLUDE_DIR} # nlohmann/json.hpp
    )
    target_link_libraries(ai_core_config PUBLIC ai_core::ai_core)
    set_target_properties(ai_core_config PROPERTIES
        VERSION ${AI_CORE_VERSION}
        SOVERSION ${AI_CORE_VERSION_MAJOR}
    )

    install(
        TARGETS ai_core_config
        EXPORT ai_core_export_set
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
    )
    install(
        FILES api/ai_core/config/algo_config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ai_core/config
        COMPONENT dev
    )
endif()

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