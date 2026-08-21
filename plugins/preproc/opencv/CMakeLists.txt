load_opencv()

set(AI_CORE_PREPROC_OPENCV_SOURCES
    ${PROJECT_SOURCE_DIR}/plugins/common/opencv/vision_util.cpp
    ${PROJECT_SOURCE_DIR}/plugins/common/preproc/generic_frame_preproc_base.cpp
    preproc/cpu_generic_preprocessor.cpp
    preproc/frame_with_mask_prep.cpp
    register.cpp
)

add_library(ai_core_plugin_preproc_opencv SHARED ${AI_CORE_PREPROC_OPENCV_SOURCES})
add_library(ai_core::plugin_preproc_opencv ALIAS ai_core_plugin_preproc_opencv)
target_compile_features(ai_core_plugin_preproc_opencv PRIVATE cxx_std_20)
target_include_directories(ai_core_plugin_preproc_opencv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/plugins/common/include
    ${PROJECT_SOURCE_DIR}/plugins/common/opencv
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ai_core_plugin_preproc_opencv PRIVATE
    ai_core::ai_core
    ${OpenCV_LIBS}
)
set_target_properties(ai_core_plugin_preproc_opencv PROPERTIES
    VERSION ${AI_CORE_VERSION}
    SOVERSION ${AI_CORE_VERSION_MAJOR}
)
install(TARGETS ai_core_plugin_preproc_opencv
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
)
