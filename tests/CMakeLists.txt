cmake_minimum_required(VERSION 3.18)

project(AICoreTests LANGUAGES CXX)

include("${AI_CORE_CMAKE_DIR}/dependencies/DevTools.cmake")
ai_core_load_googletest()

file(MAKE_DIRECTORY "${PROJECT_OUTPUT_DIR}/bin/tests")

function(ai_core_add_test_suite target label)
    add_executable(${target} ${ARGN})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE ai_core::ai_core GTest::gtest_main)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_OUTPUT_DIR}/bin/tests")
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
        LABELS ${label}
        WORKING_DIRECTORY "${AI_CORE_SOURCE_DIR}")
    install(TARGETS ${target} DESTINATION tests COMPONENT tests)
endfunction()

# The core suite deliberately uses only ai_core and GoogleTest. In particular,
# enabling tests no longer causes OpenCV or an inference SDK to be discovered.
ai_core_add_test_suite(ai_core_core_tests core
    test_core_types.cpp
    test_runtime.cpp
    test_tensor_data.cpp
    test_typed_buffer.cpp
)

if(TARGET ai_core_config)
    ai_core_add_test_suite(ai_core_config_tests config test_config.cpp)
    target_link_libraries(ai_core_config_tests PRIVATE ai_core::config)
endif()

set(AI_CORE_HAVE_VISION_TEST_COMPONENTS OFF)
if(TARGET ai_core_plugin_preproc_opencv AND
   TARGET ai_core_plugin_postproc_opencv)
    set(AI_CORE_HAVE_VISION_TEST_COMPONENTS ON)

    include("${AI_CORE_CMAKE_DIR}/dependencies/OpenCV.cmake")
    include("${AI_CORE_CMAKE_DIR}/dependencies/Platform.cmake")
    ai_core_load_opencv()
    ai_core_load_platform()

    add_library(ai_core_test_bad_version_plugin SHARED
        plugins/bad_version_plugin.cpp)
    target_compile_features(ai_core_test_bad_version_plugin PRIVATE cxx_std_20)
    target_link_libraries(ai_core_test_bad_version_plugin PRIVATE
        ai_core::ai_core)
    set_target_properties(ai_core_test_bad_version_plugin PROPERTIES
        OUTPUT_NAME ai_core_plugin_test_bad_version
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_OUTPUT_DIR}/lib/test-plugins"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_OUTPUT_DIR}/bin/test-plugins")

    ai_core_add_test_suite(ai_core_vision_tests vision
        main.cc
        test_image_view.cpp
        test_plugin_manager.cpp
        test_postproc_decode.cpp
        test_preproc_cpu.cpp
        test_vision_util.cpp
    )
    target_include_directories(ai_core_vision_tests PRIVATE
        "${AI_CORE_SOURCE_DIR}/plugins/common/include"
        "${AI_CORE_SOURCE_DIR}/plugins/common/opencv"
        "${AI_CORE_SOURCE_DIR}/plugins/preproc/opencv"
        "${AI_CORE_SOURCE_DIR}/plugins/postproc/opencv"
    )
    target_link_libraries(ai_core_vision_tests PRIVATE
        ai_core::plugin_preproc_opencv
        ai_core::plugin_postproc_opencv
        ai_core::opencv_imgcodecs
        ai_core::platform
    )
    add_dependencies(ai_core_vision_tests ai_core_test_bad_version_plugin)
    target_compile_definitions(ai_core_vision_tests PRIVATE
        "AI_CORE_TEST_PREPROC_PLUGIN=\"$<TARGET_FILE:ai_core_plugin_preproc_opencv>\""
        "AI_CORE_TEST_POSTPROC_PLUGIN=\"$<TARGET_FILE:ai_core_plugin_postproc_opencv>\""
        "AI_CORE_TEST_BAD_VERSION_PLUGIN=\"$<TARGET_FILE:ai_core_test_bad_version_plugin>\""
    )
endif()

set(AI_CORE_BACKEND_TEST_SOURCES
    main.cc
    test_algo_infer.cpp
    test_ocr_det.cpp
    test_ocr_reco.cpp
    test_yolo_det.cpp
)
if(TARGET ai_core_plugin_tensorrt)
    list(APPEND AI_CORE_BACKEND_TEST_SOURCES test_trt_infer.cpp)
endif()

if(AI_CORE_HAVE_VISION_TEST_COMPONENTS AND
   (TARGET ai_core_plugin_onnxruntime OR
    TARGET ai_core_plugin_ncnn OR
    TARGET ai_core_plugin_tensorrt))
    ai_core_add_test_suite(ai_core_backend_tests backend
        ${AI_CORE_BACKEND_TEST_SOURCES})
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
    target_include_directories(ai_core_backend_tests PRIVATE
        "${AI_CORE_SOURCE_DIR}/src"
        "${AI_CORE_SOURCE_DIR}/plugins/common/include"
        "${AI_CORE_SOURCE_DIR}/plugins/common/opencv"
        "${AI_CORE_SOURCE_DIR}/plugins/preproc/opencv"
        "${AI_CORE_SOURCE_DIR}/plugins/postproc/opencv"
        "${AI_CORE_SOURCE_DIR}/plugins/infer/onnxruntime"
        "${AI_CORE_SOURCE_DIR}/plugins/infer/ncnn"
        "${AI_CORE_SOURCE_DIR}/plugins/infer/tensorrt"
    )
    target_link_libraries(ai_core_backend_tests PRIVATE
        ai_core::plugin_preproc_opencv
        ai_core::plugin_postproc_opencv
        ai_core::opencv_imgcodecs
        ai_core::platform
    )
    target_compile_definitions(ai_core_backend_tests PRIVATE
        "AI_CORE_TEST_PREPROC_PLUGIN=\"$<TARGET_FILE:ai_core_plugin_preproc_opencv>\""
        "AI_CORE_TEST_POSTPROC_PLUGIN=\"$<TARGET_FILE:ai_core_plugin_postproc_opencv>\""
        "AI_CORE_TEST_OUTPUT_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/artifacts\""
    )
    if(AI_CORE_ENABLE_MODEL_DECRYPTION)
        target_compile_definitions(ai_core_backend_tests PRIVATE
            AI_CORE_TEST_MODEL_DECRYPTION=1)
    endif()

    if(TARGET ai_core_plugin_onnxruntime)
        target_link_libraries(ai_core_backend_tests PRIVATE
            onnxruntime::onnxruntime ai_core::plugin_onnxruntime)
        target_compile_definitions(ai_core_backend_tests PRIVATE
            WITH_ORT
            "AI_CORE_TEST_ORT_PLUGIN=\"$<TARGET_FILE:ai_core_plugin_onnxruntime>\"")
    endif()
    if(TARGET ai_core_plugin_ncnn)
        target_link_libraries(ai_core_backend_tests PRIVATE
            ai_core::ncnn ai_core::plugin_ncnn)
        target_compile_definitions(ai_core_backend_tests PRIVATE
            WITH_NCNN
            "AI_CORE_TEST_NCNN_PLUGIN=\"$<TARGET_FILE:ai_core_plugin_ncnn>\"")
    endif()
    if(TARGET ai_core_plugin_tensorrt)
        target_link_libraries(ai_core_backend_tests PRIVATE
            TensorRT::nvinfer CUDA::cudart_static
            ai_core::plugin_tensorrt)
        target_compile_definitions(ai_core_backend_tests PRIVATE
            WITH_TRT
            "AI_CORE_TEST_TRT_PLUGIN=\"$<TARGET_FILE:ai_core_plugin_tensorrt>\"")
    endif()
elseif(TARGET ai_core_plugin_onnxruntime OR
       TARGET ai_core_plugin_ncnn OR
       TARGET ai_core_plugin_tensorrt)
    message(STATUS
        "Backend integration tests skipped: enable both OpenCV plugins")
endif()
