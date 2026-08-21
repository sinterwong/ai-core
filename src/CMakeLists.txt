cmake_minimum_required(VERSION 3.18)

project(AICoreSrc VERSION ${AI_CORE_VERSION} LANGUAGES CXX)

# Enable position independent code for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Android-specific configuration
if(TARGET_OS STREQUAL "Android")
    load_android_env()
    list(APPEND ANDROID_JIN_LIBS mediandk)
endif()

# Base dependencies
set(DEPENDENCY_INCLUDES)

set(DEPENDENCY_LIBS ${ANDROID_JIN_LIBS})

# Public API headers live outside the implementation tree so consumers cannot
# accidentally depend on private headers.
set(AI_CORE_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
file(GLOB_RECURSE AI_CORE_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${AI_CORE_PUBLIC_INCLUDE_DIR}/ai_core/*.hpp"
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
    logger.cpp
    infer_base.cpp
    plugin_registry.cpp
    plugin_manager.cpp
)

# Private headers are only used for IDE indexing; globbing them is harmless.
file(GLOB AI_CORE_PRIVATE_HEADERS CONFIGURE_DEPENDS "*.hpp")

add_library(ai_core SHARED
    ${AI_CORE_SOURCES}
    ${AI_CORE_PRIVATE_HEADERS}
    ${AI_CORE_PUBLIC_HEADERS}
)

# Create namespaced alias for modern CMake usage
add_library(ai_core::ai_core ALIAS ai_core)

# Set C++ standard using target property (modern CMake style)
target_compile_features(ai_core PUBLIC cxx_std_20)

# Configure include directories
target_include_directories(ai_core
    PUBLIC
        $<BUILD_INTERFACE:${AI_CORE_PUBLIC_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DEPENDENCY_INCLUDES}
)

# Link dependencies
target_link_libraries(ai_core
    PRIVATE ${DEPENDENCY_LIBS} ${CMAKE_DL_LIBS}
)

# ============================================================================
# Compile Definitions
# ============================================================================
# Define macros based on enabled engines
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
file(GLOB AI_CORE_PUBLIC_HEADERS_TOP CONFIGURE_DEPENDS
    "${AI_CORE_PUBLIC_INCLUDE_DIR}/ai_core/*.hpp")
install(
    FILES ${AI_CORE_PUBLIC_HEADERS_TOP}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ai_core
    COMPONENT dev
)

# ============================================================================
# Optional module: ai_core::config (JSON pipeline loader)
# ============================================================================
if(AI_CORE_BUILD_CONFIG)
    message(STATUS "Configuring config module (ai_core::config)")
    add_library(ai_core_config SHARED config/algo_config.cpp)
    add_library(ai_core::config ALIAS ai_core_config)
    target_compile_features(ai_core_config PUBLIC cxx_std_20)
    target_include_directories(ai_core_config
        PUBLIC
            $<BUILD_INTERFACE:${AI_CORE_PUBLIC_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${PROJECT_INCLUDE_DIR} # nlohmann/json.hpp
    )
    target_link_libraries(ai_core_config PUBLIC ai_core::ai_core)
    # EXPORT_NAME makes the installed target `ai_core::config` too, matching the
    # build-tree ALIAS above — consumers use one name either way.
    set_target_properties(ai_core_config PROPERTIES
        EXPORT_NAME config
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
        FILES ${AI_CORE_PUBLIC_INCLUDE_DIR}/ai_core/config/algo_config.hpp
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

# Package config lands in the conventional lib/cmake/<name> location, so
# find_package(ai_core) works from a plain CMAKE_PREFIX_PATH pointing at the
# install prefix — no need for consumers to append <prefix>/share by hand.
set(AI_CORE_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/ai_core)

# configure_package_config_file (not configure_file) is what expands
# @PACKAGE_INIT@ and makes the generated config relocatable.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai_core-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config.cmake"
    INSTALL_DESTINATION ${AI_CORE_CMAKE_INSTALL_DIR}
)

# Install package configuration files
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ai_core-config-version.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/../cmake/AICorePlugin.cmake"
    DESTINATION ${AI_CORE_CMAKE_INSTALL_DIR}
    COMPONENT dev
)

# Install targets for install tree
install(
    EXPORT ai_core_export_set
    FILE ai_core-targets.cmake
    NAMESPACE ai_core::
    DESTINATION ${AI_CORE_CMAKE_INSTALL_DIR}
    COMPONENT dev
)
