cmake_minimum_required(VERSION 3.18)

project(AICoreExamplesOCRReco VERSION ${AI_CORE_VERSION})

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Load Dependencies
# ============================================================================
load_opencv()

# ============================================================================
# Source Files
# ============================================================================
set(OCR_EXAMPLE_SOURCES
    ocr_main.cc
    ocr_rec.cpp
    ocr_utils.cpp
)

# ============================================================================
# Dependencies Configuration
# ============================================================================
set(DEPENDENCY_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(DEPENDENCY_LIBS
    ai_core::ai_core_examples
    ai_core::ai_core
    ${OpenCV_LIBS}
)

# ============================================================================
# OCR Example Executable
# ============================================================================
add_executable(ai_core_example_ocr ${OCR_EXAMPLE_SOURCES})

# Set C++ standard
target_compile_features(ai_core_example_ocr PRIVATE cxx_std_20)

# Configure include directories
target_include_directories(ai_core_example_ocr PRIVATE ${DEPENDENCY_INCLUDES})

# Link dependencies
target_link_libraries(ai_core_example_ocr PRIVATE ${DEPENDENCY_LIBS})

# ============================================================================
# Installation
# ============================================================================
install(
    TARGETS ai_core_example_ocr
    EXPORT ai_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)
