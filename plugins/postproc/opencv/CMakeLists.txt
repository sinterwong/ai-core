load_opencv()

set(AI_CORE_POSTPROC_OPENCV_SOURCES
    ${PROJECT_SOURCE_DIR}/src/vision_util.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/yolo_det.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/nano_det.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/rtm_det.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/softmax_cls.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/argmax_cls.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/fpr_cls.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/semantic_seg.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/unet_dual_out_seg.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/ocr_reco.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc/raw_output.cpp
    register.cpp
)

add_library(ai_core_plugin_postproc_opencv SHARED
    ${AI_CORE_POSTPROC_OPENCV_SOURCES})
add_library(ai_core::plugin_postproc_opencv ALIAS
    ai_core_plugin_postproc_opencv)
target_compile_features(ai_core_plugin_postproc_opencv PRIVATE cxx_std_20)
target_include_directories(ai_core_plugin_postproc_opencv PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ai_core_plugin_postproc_opencv PRIVATE
    ai_core::ai_core
    ${OpenCV_LIBS}
)
set_target_properties(ai_core_plugin_postproc_opencv PROPERTIES
    VERSION ${AI_CORE_VERSION}
    SOVERSION ${AI_CORE_VERSION_MAJOR}
)
install(TARGETS ai_core_plugin_postproc_opencv
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
)
