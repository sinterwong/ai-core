include("${AI_CORE_CMAKE_DIR}/dependencies/InferenceSDKs.cmake")
include("${AI_CORE_CMAKE_DIR}/dependencies/OpenCV.cmake")
ai_core_load_onnxruntime()
ai_core_load_opencv()

add_library(ai_core_plugin_onnxruntime SHARED
    ort/ort_infer.cpp
    register.cpp
)
add_library(ai_core::plugin_onnxruntime ALIAS ai_core_plugin_onnxruntime)
target_compile_features(ai_core_plugin_onnxruntime PRIVATE cxx_std_20)
target_include_directories(ai_core_plugin_onnxruntime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ai_core_plugin_onnxruntime PRIVATE
    ai_core::ai_core
    onnxruntime::onnxruntime
    ai_core::opencv
)
if(AI_CORE_ENABLE_MODEL_DECRYPTION)
    target_compile_definitions(ai_core_plugin_onnxruntime PRIVATE
        AI_CORE_ENABLE_MODEL_DECRYPTION=1)
    target_link_libraries(ai_core_plugin_onnxruntime PRIVATE encrypt::encrypt)
endif()
install(TARGETS ai_core_plugin_onnxruntime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
)
