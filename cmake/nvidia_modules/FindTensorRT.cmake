if(NOT TRT_ROOT AND DEFINED ENV{TRT_ROOT})
    set(TRT_ROOT "$ENV{TRT_ROOT}")
endif()

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${TRT_ROOT} /usr/local/include /opt/TensorRT
    PATH_SUFFIXES include
)

find_library(TensorRT_nvinfer_LIBRARY nvinfer
    HINTS ${TRT_ROOT} /usr/local/lib /opt/TensorRT
    PATH_SUFFIXES lib lib64
)
find_library(TensorRT_nvinfer_plugin_LIBRARY nvinfer_plugin
    HINTS ${TRT_ROOT} /usr/local/lib /opt/TensorRT
    PATH_SUFFIXES lib lib64
)
find_library(TensorRT_nvonnxparser_LIBRARY nvonnxparser
    HINTS ${TRT_ROOT} /usr/local/lib /opt/TensorRT
    PATH_SUFFIXES lib lib64
)

set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
set(TensorRT_LIBRARIES
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
    ${TensorRT_nvinfer_plugin_LIBRARY}
)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" tensorrt_version_str
         REGEX "^#define NV_TENSORRT_VERSION_((MAJOR)|(MINOR)|(PATCH)|(BUILD)) .*")
    string(REGEX REPLACE ".*#define NV_TENSORRT_VERSION_MAJOR ([0-9]+).*" "\\1" NV_TENSORRT_MAJOR "${tensorrt_version_str}")
    string(REGEX REPLACE ".*#define NV_TENSORRT_VERSION_MINOR ([0-9]+).*" "\\1" NV_TENSORRT_MINOR "${tensorrt_version_str}")
    string(REGEX REPLACE ".*#define NV_TENSORRT_VERSION_PATCH ([0-9]+).*" "\\1" NV_TENSORRT_PATCH "${tensorrt_version_str}")
    set(TensorRT_VERSION "${NV_TENSORRT_MAJOR}.${NV_TENSORRT_MINOR}.${NV_TENSORRT_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    FOUND_VAR TensorRT_FOUND
    REQUIRED_VARS TensorRT_LIBRARIES TensorRT_INCLUDE_DIRS
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer INTERFACE IMPORTED)
    target_include_directories(TensorRT::nvinfer INTERFACE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(TensorRT::nvinfer INTERFACE ${TensorRT_LIBRARIES})
endif()
