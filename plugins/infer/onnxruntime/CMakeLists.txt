load_onnxruntime()
load_opencv()

add_library(ai_core_plugin_onnxruntime SHARED
    ${PROJECT_SOURCE_DIR}/src/ort/ort_infer.cpp
    register.cpp
)
add_library(ai_core::plugin_onnxruntime ALIAS ai_core_plugin_onnxruntime)
target_compile_features(ai_core_plugin_onnxruntime PRIVATE cxx_std_20)
target_include_directories(ai_core_plugin_onnxruntime PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ai_core_plugin_onnxruntime PRIVATE
    ai_core::ai_core
    onnxruntime::onnxruntime
    encrypt::encrypt
    ${OpenCV_LIBS}
)
install(TARGETS ai_core_plugin_onnxruntime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
)
