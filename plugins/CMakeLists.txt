cmake_minimum_required(VERSION 3.18)

if(AI_CORE_ENABLE_MODEL_DECRYPTION AND
   (AI_CORE_BUILD_PLUGIN_ONNXRUNTIME OR
    AI_CORE_BUILD_PLUGIN_NCNN OR
    AI_CORE_BUILD_PLUGIN_TENSORRT))
    include("${AI_CORE_CMAKE_DIR}/dependencies/Encryption.cmake")
    ai_core_load_encryption()
endif()

if(AI_CORE_BUILD_PLUGIN_PREPROC_OPENCV)
    add_subdirectory(preproc/opencv)
endif()

if(AI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV)
    add_subdirectory(postproc/opencv)
endif()

if(AI_CORE_BUILD_PLUGIN_ONNXRUNTIME)
    add_subdirectory(infer/onnxruntime)
endif()

if(AI_CORE_BUILD_PLUGIN_NCNN)
    add_subdirectory(infer/ncnn)
endif()

if(AI_CORE_BUILD_PLUGIN_TENSORRT)
    add_subdirectory(infer/tensorrt)
endif()
