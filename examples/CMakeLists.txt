cmake_minimum_required(VERSION 3.18)

project(AICoreExamples VERSION ${AI_CORE_VERSION})

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Security key for examples
set(COMMIT_CODE "689bc3e3bdf1c5f2cff81725011ba7d3c0089b25")

include("${AI_CORE_CMAKE_DIR}/dependencies/OpenCV.cmake")
include("${AI_CORE_CMAKE_DIR}/dependencies/Platform.cmake")
ai_core_load_opencv()
ai_core_load_platform()

# ============================================================================
# Source Files
# ============================================================================
set(EXAMPLES_SOURCES
    generic_image_infer.cpp
)

# ============================================================================
# Dependencies Configuration
# ============================================================================
set(DEPENDENCY_INCLUDES
    ${PROJECT_INCLUDE_DIR}
)

set(DEPENDENCY_LIBS
    ai_core::ai_core
    ai_core::config
    ai_core::opencv
    ai_core::platform
)

# ============================================================================
# Examples Library (Static)
# ============================================================================
add_library(ai_core_examples STATIC ${EXAMPLES_SOURCES})

# Create namespaced alias
add_library(ai_core::ai_core_examples ALIAS ai_core_examples)

# Set C++ standard
target_compile_features(ai_core_examples PUBLIC cxx_std_20)

# Configure include directories
target_include_directories(ai_core_examples
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        ${DEPENDENCY_INCLUDES}
)

# Link dependencies
target_link_libraries(ai_core_examples PRIVATE ${DEPENDENCY_LIBS})

# Add security key definition
target_compile_definitions(ai_core_examples PRIVATE SECURITY_KEY="${COMMIT_CODE}")

# ============================================================================
# Installation
# ============================================================================
install(
    TARGETS ai_core_examples
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT examples
)

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(ocr)
add_subdirectory(async_pipeline)
