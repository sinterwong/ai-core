cmake_minimum_required(VERSION 3.18)

if(AI_CORE_PLUGIN_VISION)
    add_subdirectory(preproc/opencv)
    add_subdirectory(postproc/opencv)
endif()

# Inference engines are plugins too. Their existing options remain available
# during the directory migration; no backend object is ever folded into core.
if(WITH_ORT_ENGINE)
    add_subdirectory(infer/onnxruntime)
endif()

if(WITH_NCNN_ENGINE)
    add_subdirectory(infer/ncnn)
endif()

if(WITH_TRT_ENGINE)
    add_subdirectory(infer/tensorrt)
endif()
