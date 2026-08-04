load_openmp()
load_ncnn()
load_opencv()

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
    ${NCNN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ai_core_plugin_ncnn PRIVATE
    ai_core::ai_core
    encrypt::encrypt
    ${NCNN_LIBS}
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)
install(TARGETS ai_core_plugin_ncnn
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
)
