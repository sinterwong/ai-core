include("${AI_CORE_CMAKE_DIR}/dependencies/InferenceSDKs.cmake")
include("${AI_CORE_CMAKE_DIR}/dependencies/OpenCV.cmake")
ai_core_load_ncnn()
ai_core_load_opencv()

add_library(ai_core_plugin_ncnn SHARED
    ncnn/ncnn_infer.cpp
    ncnn/ncnn_generic_preprocessor.cpp
    register.cpp
)
add_library(ai_core::plugin_ncnn ALIAS ai_core_plugin_ncnn)
target_compile_features(ai_core_plugin_ncnn PRIVATE cxx_std_20)
target_include_directories(ai_core_plugin_ncnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/plugins/common/include
)
target_link_libraries(ai_core_plugin_ncnn PRIVATE
    ai_core::ai_core
    ai_core::ncnn
    ai_core::opencv
)
if(AI_CORE_ENABLE_MODEL_DECRYPTION)
    target_compile_definitions(ai_core_plugin_ncnn PRIVATE
        AI_CORE_ENABLE_MODEL_DECRYPTION=1)
    target_link_libraries(ai_core_plugin_ncnn PRIVATE encrypt::encrypt)
endif()
set_target_properties(ai_core_plugin_ncnn PROPERTIES
    EXPORT_NAME plugin_ncnn
    VERSION ${AI_CORE_VERSION}
    SOVERSION ${AI_CORE_VERSION_MAJOR}
)
install(TARGETS ai_core_plugin_ncnn
    EXPORT ai_core_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
        COMPONENT plugin-ncnn
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
        COMPONENT plugin-ncnn
)
