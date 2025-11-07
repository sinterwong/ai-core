cmake_minimum_required(VERSION 3.18)
if(WITH_TRT_ENGINE)
    project(AICoreSrc VERSION ${AI_CORE_VERSION} LANGUAGES CXX CUDA)
else()
    project(AICoreSrc VERSION ${AI_CORE_VERSION} LANGUAGES CXX)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

load_opencv()

if(TARGET_OS STREQUAL "Android")
    load_android_env()
    list(APPEND ANDROID_JIN_LIBS mediandk)
endif()

set(DEPENDENCY_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
)

set(DEPENDENCY_LIBS
    encrypt::encrypt
    logger::logger
    ${OpenCV_LIBS}
    ${ANDROID_JIN_LIBS}
)

# Public headers
file(GLOB_RECURSE CURRENT_DIR_PUBLIC_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/api/*.hpp)

# Private headers
file(GLOB CURRENT_DIR_PRIVATE_HEAD
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/postproc/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/preproc/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/registrar/*.hpp
)

file(GLOB CURRENT_DIR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/postproc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/preproc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/registrar/*.cpp
)

if(WITH_ORT_ENGINE)
    load_onnxruntime()
    add_subdirectory(ort)
    list(APPEND DEPENDENCY_LIBS onnxruntime::onnxruntime)
endif()

if(WITH_NCNN_ENGINE)
    load_openmp()
    load_ncnn()
    add_subdirectory(ncnn)

    list(APPEND DEPENDENCY_INCLUDES
        ${NCNN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn
    )
    list(APPEND DEPENDENCY_LIBS ${NCNN_MODULE_LIBS})
endif()

if(WITH_TRT_ENGINE)
    load_cuda()
    load_tensorrt()

    set(CMAKE_CUDA_ARCHITECTURES 87 86)

    add_subdirectory(trt)

    list(APPEND DEPENDENCY_INCLUDES
        ${TensorRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trt
    )

    list(APPEND DEPENDENCY_LIBS ${TRT_MODULE_LIBS})
endif()

add_library(ai_core SHARED
    ${CURRENT_DIR_SRCS}
    ${CURRENT_DIR_PRIVATE_HEAD}

    # if WITH_ORT_ENGINE is ON
    $<$<BOOL:${WITH_ORT_ENGINE}>:$<TARGET_OBJECTS:ai_ort_module>>
    # if WITH_NCNN_ENGINE is ON
    $<$<BOOL:${WITH_NCNN_ENGINE}>:$<TARGET_OBJECTS:ai_ncnn_module>>
    # if WITH_TRT_ENGINE is ON
    $<$<BOOL:${WITH_TRT_ENGINE}>:$<TARGET_OBJECTS:ai_trt_module>>
)
add_library(ai_core::ai_core ALIAS ai_core)

if(WITH_TRT_ENGINE)
    message(STATUS "Applying usage requirements from ai_trt_module to ai_core.")
    # Apply target properties and compile options from trt submodule
    set_target_properties(ai_core PROPERTIES ${AI_TRT_CONSUMER_PROPERTIES})
    target_compile_options(ai_core PRIVATE ${AI_TRT_CONSUMER_COMPILE_OPTIONS})
endif()

target_include_directories(ai_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${DEPENDENCY_INCLUDES}
)

target_link_libraries(ai_core PRIVATE ${DEPENDENCY_LIBS})

# Add compile definitions
if(WITH_ORT_ENGINE)
    target_compile_definitions(ai_core PRIVATE WITH_ORT)
endif()

if(WITH_NCNN_ENGINE)
    target_compile_definitions(ai_core PRIVATE WITH_NCNN)
endif()

if(WITH_TRT_ENGINE)
    target_compile_definitions(ai_core PRIVATE WITH_TRT)
endif()

install(TARGETS ai_core
    EXPORT ai_export_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)

install(FILES ${CURRENT_DIR_PUBLIC_HEAD}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ai_core
    COMPONENT dev
)

# Generate the ai_core-config.cmake and ai_core-config-version.cmake files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT ai_export_export_set
    FILE "${CMAKE_CURRENT_BINARY_DIR}/ai_core-targets.cmake"
    NAMESPACE ai_core::
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai_core-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_DATADIR}
    COMPONENT dev
)

install(EXPORT ai_export_export_set
    FILE ai_core-targets.cmake
    NAMESPACE ai_core::
    DESTINATION ${CMAKE_INSTALL_DATADIR}
    COMPONENT dev
)
