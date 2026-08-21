enable_language(CUDA)
load_cuda()
load_tensorrt()
load_opencv()

if(AI_CORE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${AI_CORE_CUDA_ARCHITECTURES})
else()
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(ai_core_plugin_tensorrt SHARED
    ${PROJECT_SOURCE_DIR}/plugins/common/preproc/generic_frame_preproc_base.cpp
    trt/trt_infer.cpp
    trt/trt_utils.cpp
    trt/trt_infer_stream.cpp
    trt/cuda_utils.cu
    trt/cuda_accelerator_buffer_impl.cu
    trt/gpu_generic_cuda_preprocessor.cu
    register.cpp
)
add_library(ai_core::plugin_tensorrt ALIAS ai_core_plugin_tensorrt)
target_compile_features(ai_core_plugin_tensorrt PRIVATE cxx_std_20)
set_target_properties(ai_core_plugin_tensorrt PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_definitions(ai_core_plugin_tensorrt PRIVATE WITH_TRT=1)
target_include_directories(ai_core_plugin_tensorrt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/trt
    ${PROJECT_SOURCE_DIR}/plugins/common/include
    ${TensorRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ai_core_plugin_tensorrt PRIVATE
    ai_core::ai_core
    TensorRT::nvinfer
    CUDA::cudart_static
    encrypt::encrypt
    ${OpenCV_LIBS}
)
install(TARGETS ai_core_plugin_tensorrt
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ai_core/plugins
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ai_core/plugins
)
