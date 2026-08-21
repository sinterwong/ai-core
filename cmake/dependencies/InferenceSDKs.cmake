include_guard(GLOBAL)
include("${AI_CORE_CMAKE_DIR}/dependencies/Common.cmake")

set(AI_CORE_ONNXRUNTIME_ROOT "${AI_CORE_DEPS_ROOT}/onnxruntime" CACHE PATH
    "ONNX Runtime SDK root")
set(AI_CORE_NCNN_ROOT "${AI_CORE_DEPS_ROOT}/ncnn" CACHE PATH "NCNN SDK root")
set(AI_CORE_TENSORRT_ROOT "${AI_CORE_DEPS_ROOT}/tensorrt" CACHE PATH
    "TensorRT SDK root")

function(ai_core_load_onnxruntime)
    if(TARGET onnxruntime::onnxruntime)
        return()
    endif()
    ai_core_require_directory(
        "ONNX Runtime" "${AI_CORE_ONNXRUNTIME_ROOT}" "onnxruntime")
    find_package(onnxruntime CONFIG REQUIRED
        PATHS "${AI_CORE_ONNXRUNTIME_ROOT}/lib/cmake/onnxruntime"
        NO_DEFAULT_PATH)
    # find_package() is invoked from a component directory. Promote imported
    # SDK targets so sibling developer targets (tests/benchmarks) can consume
    # the same resolved dependency without running discovery again.
    set_property(TARGET onnxruntime::onnxruntime PROPERTY IMPORTED_GLOBAL TRUE)
endfunction()

function(ai_core_load_ncnn)
    if(TARGET ai_core::ncnn)
        return()
    endif()
    ai_core_require_directory("NCNN" "${AI_CORE_NCNN_ROOT}" "ncnn")
    find_package(ncnn CONFIG REQUIRED
        PATHS
            "${AI_CORE_NCNN_ROOT}"
            "${AI_CORE_NCNN_ROOT}/lib/cmake/ncnn"
        NO_DEFAULT_PATH)
    get_target_property(ncnn_is_imported ncnn IMPORTED)
    if(ncnn_is_imported)
        set_property(TARGET ncnn PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
    find_package(OpenMP REQUIRED COMPONENTS CXX)

    add_library(ai_core_ncnn INTERFACE)
    add_library(ai_core::ncnn ALIAS ai_core_ncnn)
    target_link_libraries(ai_core_ncnn INTERFACE ncnn OpenMP::OpenMP_CXX)
endfunction()

function(ai_core_load_tensorrt)
    if(TARGET TensorRT::nvinfer)
        return()
    endif()
    ai_core_require_directory(
        "TensorRT" "${AI_CORE_TENSORRT_ROOT}" "tensorrt")
    find_package(CUDAToolkit REQUIRED)
    set(TRT_ROOT "${AI_CORE_TENSORRT_ROOT}")
    list(APPEND CMAKE_MODULE_PATH "${AI_CORE_CMAKE_DIR}/nvidia_modules")
    find_package(TensorRT REQUIRED)
    foreach(target TensorRT::nvinfer TensorRT::nvinfer_plugin
                   TensorRT::nvonnxparser CUDA::cudart_static)
        if(TARGET ${target})
            set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL TRUE)
        endif()
    endforeach()
endfunction()
