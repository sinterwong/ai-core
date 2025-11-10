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
    logger::logger
    ${OpenCV_LIBS}
    ${ANDROID_JIN_LIBS}
)

# ============================================================================
# Source Files - Explicitly Listed
# ============================================================================
# Public API headers
set(AI_CORE_PUBLIC_HEADERS
    api/ai_core/ai_core_registrar.hpp
    api/ai_core/ai_core_version.hpp
    api/ai_core/algo_data_types.hpp
    api/ai_core/algo_infer.hpp
    api/ai_core/algo_infer_engine.hpp
    api/ai_core/algo_input_types.hpp
    api/ai_core/algo_manager.hpp
    api/ai_core/algo_output_types.hpp
    api/ai_core/algo_postproc.hpp
    api/ai_core/algo_preproc.hpp
    api/ai_core/data_packet.hpp
    api/ai_core/device_buffer_impl.hpp
    api/ai_core/infer_base.hpp
    api/ai_core/infer_common_types.hpp
    api/ai_core/infer_error_code.hpp
    api/ai_core/infer_params_types.hpp
    api/ai_core/param_center.hpp
    api/ai_core/postproc_base.hpp
    api/ai_core/postproc_types.hpp
    api/ai_core/preproc_base.hpp
    api/ai_core/preproc_types.hpp
    api/ai_core/tensor_data.hpp
    api/ai_core/type_safe_factory.hpp
    api/ai_core/typed_buffer.hpp
)

# Private implementation headers
set(AI_CORE_PRIVATE_HEADERS
    # Core headers
    algo_infer_engine_impl.hpp
    algo_infer_impl.hpp
    algo_manager_impl.hpp
    algo_postproc_impl.hpp
    algo_preproc_impl.hpp
    vision_util.hpp

    # Preprocessing headers
    preproc/cpu_generic_preprocessor.hpp
    preproc/frame_prep.hpp
    preproc/frame_preprocessor_base.hpp
    preproc/frame_with_mask_prep.hpp

    # Postprocessing headers
    postproc/anchor_det_post_base.hpp
    postproc/anchor_det_postproc.hpp
    postproc/confidence_filter_post_base.hpp
    postproc/confidence_filter_postproc.hpp
    postproc/cv_generic_post_base.hpp
    postproc/cv_generic_postproc.hpp
    postproc/fpr_cls.hpp
    postproc/nano_det.hpp
    postproc/ocr_reco.hpp
    postproc/raw_output.hpp
    postproc/rtm_det.hpp
    postproc/semantic_seg.hpp
    postproc/softmax_cls.hpp
    postproc/unet_daul_out_seg.hpp
    postproc/yolo_det.hpp

    # Registrar headers
    registrar/infer_engine_registrar.hpp
    registrar/postproc_registrar.hpp
    registrar/preproc_registrar.hpp
)

# Source files
set(AI_CORE_SOURCES
    # Core implementation
    algo_infer.cpp
    algo_infer_engine.cpp
    algo_infer_engine_impl.cpp
    algo_infer_impl.cpp
    algo_manager.cpp
    algo_manager_impl.cpp
    algo_postproc.cpp
    algo_postproc_impl.cpp
    algo_preproc.cpp
    algo_preproc_impl.cpp
    device_buffer_impl_stub.cpp
    infer_base.cpp
    typed_buffer.cpp
    vision_util.cpp

    # Preprocessing implementation
    preproc/cpu_generic_preprocessor.cpp
    preproc/frame_prep.cpp
    preproc/frame_with_mask_prep.cpp

    # Postprocessing implementation
    postproc/anchor_det_postproc.cpp
    postproc/confidence_filter_postproc.cpp
    postproc/cv_generic_postproc.cpp
    postproc/fpr_cls.cpp
    postproc/nano_det.cpp
    postproc/ocr_reco.cpp
    postproc/raw_output.cpp
    postproc/rtm_det.cpp
    postproc/semantic_seg.cpp
    postproc/softmax_cls.cpp
    postproc/unet_daul_out_seg.cpp
    postproc/yolo_det.cpp

    # Registrar implementation
    registrar/infer_engine_registrar.cpp
    registrar/postproc_registrar.cpp
    registrar/preproc_registrar.cpp
)

# ============================================================================
# Inference Engine Modules
# ============================================================================
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

    # Set CUDA architectures for TensorRT
    set(CMAKE_CUDA_ARCHITECTURES 87 86)

    add_subdirectory(trt)

    list(APPEND DEPENDENCY_INCLUDES
        ${TensorRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trt
    )
    list(APPEND DEPENDENCY_LIBS ${TRT_MODULE_LIBS})
endif()

# ============================================================================
# Main Library Target
# ============================================================================
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
target_link_libraries(ai_core PRIVATE ${DEPENDENCY_LIBS})

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
