include("${AI_CORE_CMAKE_DIR}/dependencies/OpenCV.cmake")
ai_core_load_opencv()

set(AI_CORE_POSTPROC_OPENCV_SOURCES
    ${PROJECT_SOURCE_DIR}/plugins/common/opencv/vision_util.cpp
    postproc/yolo_det.cpp
    postproc/nano_det.cpp
    postproc/rtm_det.cpp
    postproc/softmax_cls.cpp
    postproc/argmax_cls.cpp
    postproc/fpr_cls.cpp
    postproc/semantic_seg.cpp
    postproc/unet_dual_out_seg.cpp
    postproc/ocr_reco.cpp
    postproc/raw_output.cpp
    register.cpp
)

add_library(ai_core_plugin_postproc_opencv SHARED
    ${AI_CORE_POSTPROC_OPENCV_SOURCES})
add_library(ai_core::plugin_postproc_opencv ALIAS
    ai_core_plugin_postproc_opencv)
target_compile_features(ai_core_plugin_postproc_opencv PRIVATE cxx_std_20)
target_include_directories(ai_core_plugin_postproc_opencv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/plugins/common/opencv
)
target_link_libraries(ai_core_plugin_postproc_opencv PRIVATE
    ai_core::ai_core
    ai_core::opencv
)
set_target_properties(ai_core_plugin_postproc_opencv PROPERTIES
    EXPORT_NAME plugin_postproc_opencv
    VERSION ${AI_CORE_VERSION}
    SOVERSION ${AI_CORE_VERSION_MAJOR}
)
install(TARGETS ai_core_plugin_postproc_opencv
    EXPORT ai_core_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
        COMPONENT plugin-postproc-opencv
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
        COMPONENT plugin-postproc-opencv
)
