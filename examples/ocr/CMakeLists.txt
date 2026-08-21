cmake_minimum_required(VERSION 3.18)

project(AICoreExamplesOCRReco VERSION ${AI_CORE_VERSION})

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
)

set(DEPENDENCY_LIBS
    ai_core::ai_core_examples
    ai_core::ai_core
    ai_core::config
    ai_core::opencv_imgcodecs
)

if(TARGET ai_core_plugin_preproc_opencv AND
   TARGET ai_core_plugin_postproc_opencv)
    list(APPEND DEPENDENCY_LIBS
        ai_core::plugin_preproc_opencv
        ai_core::plugin_postproc_opencv)
endif()
if(TARGET ai_core_plugin_onnxruntime)
    list(APPEND DEPENDENCY_LIBS ai_core::plugin_onnxruntime)
endif()
if(TARGET ai_core_plugin_ncnn)
    list(APPEND DEPENDENCY_LIBS ai_core::plugin_ncnn)
endif()

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

if(TARGET ai_core_plugin_preproc_opencv AND
   TARGET ai_core_plugin_postproc_opencv)
    target_compile_definitions(ai_core_example_ocr PRIVATE
        AI_CORE_EXAMPLE_PREPROC_PLUGIN="$<TARGET_FILE:ai_core_plugin_preproc_opencv>"
        AI_CORE_EXAMPLE_POSTPROC_PLUGIN="$<TARGET_FILE:ai_core_plugin_postproc_opencv>")
endif()
if(TARGET ai_core_plugin_onnxruntime)
    target_compile_definitions(ai_core_example_ocr PRIVATE
        AI_CORE_EXAMPLE_ORT_PLUGIN="$<TARGET_FILE:ai_core_plugin_onnxruntime>")
endif()
if(TARGET ai_core_plugin_ncnn)
    target_compile_definitions(ai_core_example_ocr PRIVATE
        AI_CORE_EXAMPLE_NCNN_PLUGIN="$<TARGET_FILE:ai_core_plugin_ncnn>")
endif()

# ============================================================================
# Installation
# ============================================================================
install(
    TARGETS ai_core_example_ocr
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT examples
)
