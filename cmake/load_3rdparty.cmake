cmake_minimum_required(VERSION 3.18)

# Source dependencies and downloaded SDKs have deliberately separate roots.
set(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/third_party)
set(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/.deps/${TARGET_OS}_${TARGET_ARCH})
message(STATUS "AI_CORE_DEPS_ROOT: ${3RDPARTY_DIR}")

# Load OpenCV library
#
# The vendored copy under 3rdparty/target is preferred but not exclusive: the
# prefix path is appended, so a missing vendored OpenCV falls back to the system
# one. Having *both* is the dangerous case — two libopencv_core.so in one
# process means passing a cv::Mat across a library boundary is UB — so warn.
function(load_opencv)
    # A pinned source submodule is the reproducible path. OpenCV becomes part
    # of this build (no separate user installation step) while remaining a
    # private implementation detail of the plugins. Its build policy and
    # module selection live in 3rdparty/CMakeLists.txt.
    if(EXISTS "${3RDPARTY_ROOT}/plugins/opencv/CMakeLists.txt")
        if(NOT TARGET opencv_core)
            message(FATAL_ERROR
                "Bundled OpenCV was not configured by 3rdparty/CMakeLists.txt")
        endif()
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_dnn PARENT_SCOPE)
        set(OpenCV_INCLUDE_DIRS
            "${3RDPARTY_ROOT}/plugins/opencv/include"
            "${3RDPARTY_ROOT}/plugins/opencv/modules/core/include"
            "${3RDPARTY_ROOT}/plugins/opencv/modules/imgproc/include"
            "${3RDPARTY_ROOT}/plugins/opencv/modules/dnn/include"
            "${3RDPARTY_ROOT}/plugins/opencv/modules/imgcodecs/include"
            "${CMAKE_BINARY_DIR}"
            PARENT_SCOPE)
        message(STATUS "OpenCV: bundled source build (core,imgproc,dnn)")
        return()
    endif()

    set(OPENCV_HOME ${3RDPARTY_DIR}/opencv)

    # Probe by locating the file, never with find_package: a real find_package
    # here would create OpenCV's imported targets from whichever copy it found
    # first and the library would then link *that* one — causing the very
    # problem this check exists to report.
    if(EXISTS "${OPENCV_HOME}" AND NOT TARGET_OS STREQUAL "Android")
        find_file(AI_CORE_SYSTEM_OPENCV_CONFIG
            NAMES OpenCVConfig.cmake
            PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/share /usr/local/share
            PATH_SUFFIXES cmake/opencv4 opencv4 ${CMAKE_LIBRARY_ARCHITECTURE}/cmake/opencv4
            NO_DEFAULT_PATH)
        mark_as_advanced(AI_CORE_SYSTEM_OPENCV_CONFIG)
        if(AI_CORE_SYSTEM_OPENCV_CONFIG)
            message(WARNING "A vendored OpenCV (${OPENCV_HOME}) coexists with a system OpenCV (${AI_CORE_SYSTEM_OPENCV_CONFIG}). If consumers of ai_core link the system one, the process ends up with two libopencv_core.so and cv::Mat cannot safely cross that boundary. Prefer removing one of them; verify with: ldd <your binary> | grep opencv_core")
        endif()
    endif()

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
        find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc dnn)

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

    # Resolve the module relative to this file. PROJECT_SOURCE_DIR changes in
    # nested components that call project(), such as tests and benchmarks.
    list(APPEND CMAKE_MODULE_PATH
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/nvidia_modules")
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
