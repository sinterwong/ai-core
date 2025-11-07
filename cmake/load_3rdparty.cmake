cmake_minimum_required(VERSION 3.18)

# Specialized libraries can be compiled separately, soft-linked to the 3RDPARTY_DIR, and then handled independently.
set(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/3rdparty)
set(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${TARGET_OS}_${TARGET_ARCH})
message(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")

# Load OpenCV library
function(load_opencv)
    set(OPENCV_HOME ${3RDPARTY_DIR}/opencv)

    if(TARGET_OS STREQUAL "Android")
        set(OpenCV_INCLUDE_DIRS ${OPENCV_HOME}/jni/include)
        set(OpenCV_LIBRARY_DIRS ${OPENCV_HOME}/staticlibs/${ANDROID_ABI})
        set(OpenCV_3RDPARTY_LIBRARY_DIRS ${OPENCV_HOME}/3rdparty/libs/${ANDROID_ABI})

        file(GLOB OpenCV_LIBS
            "${OpenCV_LIBRARY_DIRS}/*.a"
            "${OpenCV_3RDPARTY_LIBRARY_DIRS}/*.a"
        )
        message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

        # Export to parent scope
        set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} PARENT_SCOPE)
        set(OpenCV_LIBS ${OpenCV_LIBS} PARENT_SCOPE)

    elseif(TARGET_OS STREQUAL "Windows")
        set(OpenCV_LIBRARY_DIR ${OPENCV_HOME}/build)
        list(APPEND CMAKE_PREFIX_PATH ${OpenCV_LIBRARY_DIR})
        find_package(OpenCV REQUIRED)

        if(OpenCV_INCLUDE_DIRS)
            message(STATUS "OpenCV library status:")
            message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
            message(STATUS "    libraries dir: ${OpenCV_LIBRARY_DIR}")
            message(STATUS "    libraries: ${OpenCV_LIBS}")
        else()
            message(FATAL_ERROR "OpenCV not found!")
        endif()

        # Export to parent scope
        set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} PARENT_SCOPE)
        set(OpenCV_LIBS ${OpenCV_LIBS} PARENT_SCOPE)
        set(OpenCV_LIBRARY_DIR ${OpenCV_LIBRARY_DIR} PARENT_SCOPE)

    else()
        set(OpenCV_LIBRARY_DIR ${OPENCV_HOME}/lib)
        list(APPEND CMAKE_PREFIX_PATH ${OpenCV_LIBRARY_DIR}/cmake)
        find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc highgui video videoio imgcodecs calib3d)

        if(OpenCV_INCLUDE_DIRS)
            message(STATUS "OpenCV library status:")
            message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
            message(STATUS "    libraries dir: ${OpenCV_LIBRARY_DIR}")
            message(STATUS "    libraries: ${OpenCV_LIBS}")
        else()
            message(FATAL_ERROR "OpenCV not found!")
        endif()

        # Export to parent scope
        set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} PARENT_SCOPE)
        set(OpenCV_LIBS ${OpenCV_LIBS} PARENT_SCOPE)
        set(OpenCV_LIBRARY_DIR ${OpenCV_LIBRARY_DIR} PARENT_SCOPE)
    endif()
endfunction()

# Load ONNX Runtime library
function(load_onnxruntime)
    set(ONNXRUNTIME_HOME ${3RDPARTY_DIR}/onnxruntime)
    if(TARGET_OS STREQUAL "Android")
        set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${ONNXRUNTIME_HOME}/lib/cmake)
    else()
        list(APPEND CMAKE_PREFIX_PATH ${ONNXRUNTIME_HOME}/lib/cmake)
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
    endif()

    find_package(onnxruntime REQUIRED)
    if(onnxruntime_FOUND)
        message(STATUS "Successfully found ONNX Runtime ${onnxruntime_VERSION}")
    endif()
endfunction()

# Load NCNN library
function(load_ncnn)
    set(NCNN_LOADED TRUE PARENT_SCOPE)
    set(NCNN_HOME ${3RDPARTY_DIR}/ncnn)

    message(STATUS "NCNN_HOME: ${NCNN_HOME}")

    if(TARGET_OS STREQUAL "Android")
        set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${NCNN_HOME}/lib/cmake)
    else()
        list(APPEND CMAKE_PREFIX_PATH ${NCNN_HOME})
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
    endif()

    find_package(ncnn REQUIRED)

    if(ncnn_FOUND)
        get_target_property(NCNN_INCLUDE_DIR ncnn INTERFACE_INCLUDE_DIRECTORIES)
        set(NCNN_LIBS ncnn)
        message(STATUS "NCNN library status:")
        message(STATUS "    include path: ${NCNN_INCLUDE_DIR}")
        message(STATUS "    libraries: ${NCNN_LIBS}")

        # Export to parent scope
        set(NCNN_INCLUDE_DIR ${NCNN_INCLUDE_DIR} PARENT_SCOPE)
        set(NCNN_LIBS ${NCNN_LIBS} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "NCNN not found after calling find_package(ncnn)!")
    endif()
endfunction()

# Load OpenMP library
function(load_openmp)
    find_package(OpenMP REQUIRED)
    if(OpenMP_FOUND)
        message(STATUS "OpenMP found:")
        message(STATUS "    OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
        message(STATUS "    OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
    endif()
endfunction()

# Load CUDA toolkit
function(load_cuda)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")

    find_library(CUDART_LIB cudart_static HINTS ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib lib/x64 lib64)
    set(CUDA_LIBRARIES ${CUDART_LIB})

    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

    # Export to parent scope
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
    set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
    set(CUDAToolkit_LIBRARY_DIR ${CUDAToolkit_LIBRARY_DIR} PARENT_SCOPE)
endfunction()

# Load TensorRT library
function(load_tensorrt)
    set(TRT_ROOT ${3RDPARTY_DIR}/tensorrt)
    set(TRT_LIB_DIR ${TRT_ROOT}/lib)

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/nvidia_modules")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)

    find_package(TensorRT REQUIRED)

    message(STATUS "Successfully found TensorRT ${TensorRT_VERSION}")
endfunction()

# Load Android environment
function(load_android_env)
    set(ANDROID_JIN_INCLUDE_DIR "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include")
    set(ANDROID_JIN_LIBS_DIR "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${TARGET_ARCH}-linux-android/24")
    set(ANDROID_JIN_LIBS
        android
        log
        z
        dl
    )

    # Export to parent scope
    set(ANDROID_JIN_INCLUDE_DIR ${ANDROID_JIN_INCLUDE_DIR} PARENT_SCOPE)
    set(ANDROID_JIN_LIBS_DIR ${ANDROID_JIN_LIBS_DIR} PARENT_SCOPE)
    set(ANDROID_JIN_LIBS ${ANDROID_JIN_LIBS} PARENT_SCOPE)

    link_directories(${ANDROID_JIN_LIBS_DIR})
endfunction()
