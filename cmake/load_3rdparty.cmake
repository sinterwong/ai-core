# Specialized libraries can be compiled separately, softinked to the 3RDPARTY_DIR, and then handled independently.
SET(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/3rdparty)
SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${TARGET_OS}_${TARGET_ARCH})
MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")

MACRO(LOAD_OPENCV)
    SET(OPENCV_HOME ${3RDPARTY_DIR}/opencv)
    
    IF (TARGET_OS STREQUAL "Android")
        SET(OpenCV_INCLUDE_DIRS ${OPENCV_HOME}/jni/include)
        SET(OpenCV_LIBRARY_DIRS ${OPENCV_HOME}/staticlibs/${ANDROID_ABI})
        SET(OpenCV_3RDPARTY_LIBRARY_DIRS ${OPENCV_HOME}/3rdparty/libs/${ANDROID_ABI})

        FILE(GLOB OpenCV_LIBS
            "${OpenCV_LIBRARY_DIRS}/*.a"
            "${OpenCV_3RDPARTY_LIBRARY_DIRS}/*.a"
        )
        MESSAGE(STATUS "Opencv libraries: ${OpenCV_LIBS}")
    ELSEIF(TARGET_OS STREQUAL "Windows")
        SET(OpenCV_LIBRARY_DIR ${OPENCV_HOME}/build)
        LIST(APPEND CMAKE_PREFIX_PATH ${OpenCV_LIBRARY_DIR})
        FIND_PACKAGE(OpenCV)

        IF(OpenCV_INCLUDE_DIRS)
            MESSAGE(STATUS "Opencv library status:")
            MESSAGE(STATUS "Opencv include path: ${OpenCV_INCLUDE_DIRS}")
            MESSAGE(STATUS "Opencv libraries dir: ${OpenCV_LIBRARY_DIR}")
            MESSAGE(STATUS "Opencv libraries: ${OpenCV_LIBS}")
        ELSE()
            MESSAGE(FATAL_ERROR "OpenCV not found!")
        ENDIF()
    
        LINK_DIRECTORIES(
            ${OpenCV_LIBRARY_DIR}
        )

    ELSE()
        SET(OpenCV_LIBRARY_DIR ${OPENCV_HOME}/lib)
        LIST(APPEND CMAKE_PREFIX_PATH ${OpenCV_LIBRARY_DIR}/cmake)
        FIND_PACKAGE(OpenCV CONFIG REQUIRED COMPONENTS core imgproc highgui video videoio imgcodecs calib3d)
        
        IF(OpenCV_INCLUDE_DIRS)
            MESSAGE(STATUS "Opencv library status:")
            MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
            MESSAGE(STATUS "    libraries dir: ${OpenCV_LIBRARY_DIR}")
            MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
        ELSE()
            MESSAGE(FATAL_ERROR "OpenCV not found!")
        ENDIF()
    
        LINK_DIRECTORIES(
            ${OpenCV_LIBRARY_DIR}
        )
    ENDIF()
ENDMACRO()

MACRO(LOAD_ONNXRUNTIME)
    SET(ONNXRUNTIME_HOME ${3RDPARTY_DIR}/onnxruntime)
    SET(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_HOME}/include")
    SET(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_HOME}/lib")

    FILE(GLOB ONNXRUNTIME_LIBS
        "${ONNXRUNTIME_LIBRARY_DIR}/*.*"
    )

    IF(ONNXRUNTIME_INCLUDE_DIR)
        MESSAGE(STATUS "ONNXRUNTIME_INCLUDE_DIR : ${ONNXRUNTIME_INCLUDE_DIR}")
        MESSAGE(STATUS "ONNXRUNTIME_LIBRARY_DIR : ${ONNXRUNTIME_LIBRARY_DIR}")
        MESSAGE(STATUS "ONNXRUNTIME_LIBS : ${ONNXRUNTIME_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "ONNXRUNTIME_LIBS not found!")
    ENDIF()

    LINK_DIRECTORIES(
        ${ONNXRUNTIME_LIBRARY_DIR}
    )
ENDMACRO()

MACRO(LOAD_NCNN)
    SET(NCNN_HOME ${3RDPARTY_DIR}/ncnn)
    MESSAGE(STATUS "NCNN_HOME: ${NCNN_HOME}")

    SET(NCNN_INCLUDE_DIR "${NCNN_HOME}/include")
    SET(NCNN_LIB_DIR "${NCNN_HOME}/lib")

    SET(NCNN_LIBS
        ncnn
        MachineIndependent
        glslang
        glslang-default-resource-limits
        SPIRV
        GenericCodeGen
        OSDependent
        OGLCompiler
    )
    LINK_DIRECTORIES(${NCNN_LIB_DIR})
ENDMACRO()

MACRO(LOAD_OPENMP)
    FIND_PACKAGE(OpenMP REQUIRED)
ENDMACRO()

MACRO(LOAD_ANDROID_ENV)
    SET(ANDROID_JIN_INCLUDE_DIR "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include")
    SET(ANDROID_JIN_LIBS_DIR "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${TARGET_ARCH}-linux-android/24")
    SET(ANDROID_JIN_LIBS 
        android
        log
        z
        dl
    )
    LINK_DIRECTORIES(${ANDROID_JIN_LIBS_DIR})
ENDMACRO()
