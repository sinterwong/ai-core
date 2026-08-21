include_guard(GLOBAL)
include("${AI_CORE_CMAKE_DIR}/dependencies/Common.cmake")

set(AI_CORE_OPENCV_PROVIDER "BUNDLED" CACHE STRING
    "OpenCV provider used by official plugins: BUNDLED or SYSTEM")
set_property(CACHE AI_CORE_OPENCV_PROVIDER PROPERTY STRINGS BUNDLED SYSTEM)

function(ai_core_load_opencv)
    if(TARGET ai_core::opencv)
        return()
    endif()

    set(opencv_components core imgproc dnn)
    if(AI_CORE_BUILD_TESTS OR AI_CORE_BUILD_BENCHMARKS OR
       AI_CORE_BUILD_EXAMPLES)
        list(APPEND opencv_components imgcodecs)
    endif()

    if(AI_CORE_OPENCV_PROVIDER STREQUAL "BUNDLED")
        ai_core_require_source("OpenCV" "plugins/opencv" "vision")

        list(JOIN opencv_components "," opencv_build_list)
        set(BUILD_LIST "${opencv_build_list}" CACHE STRING
            "OpenCV modules required by ai-core" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build bundled OpenCV statically" FORCE)
        set(BUILD_TESTS OFF CACHE BOOL "Disable OpenCV tests" FORCE)
        set(BUILD_PERF_TESTS OFF CACHE BOOL "Disable OpenCV performance tests" FORCE)
        set(BUILD_opencv_apps OFF CACHE BOOL "Disable OpenCV applications" FORCE)
        set(BUILD_JAVA OFF CACHE BOOL "Disable OpenCV Java bindings" FORCE)
        set(BUILD_opencv_python3 OFF CACHE BOOL "Disable OpenCV Python bindings" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "Disable OpenCV examples" FORCE)
        set(WITH_IPP OFF CACHE BOOL "Disable downloaded IPPICV binaries" FORCE)
        set(WITH_ITT OFF CACHE BOOL "Disable Intel ITT instrumentation" FORCE)
        set(WITH_OPENCL OFF CACHE BOOL "Disable OpenCV OpenCL runtime" FORCE)
        set(WITH_OPENGL OFF CACHE BOOL "Disable OpenCV OpenGL support" FORCE)
        set(WITH_VA OFF CACHE BOOL "Disable OpenCV VA-API support" FORCE)
        set(WITH_VA_INTEL OFF CACHE BOOL "Disable Intel VA extensions" FORCE)
        set(WITH_VTK OFF CACHE BOOL "Disable OpenCV VTK support" FORCE)
        set(WITH_GTK OFF CACHE BOOL "Disable OpenCV GTK support" FORCE)

        add_subdirectory(
            "${AI_CORE_THIRD_PARTY_DIR}/plugins/opencv"
            "${CMAKE_BINARY_DIR}/_deps/opencv"
            EXCLUDE_FROM_ALL)

        set(ai_core_opencv_include_dirs
            "${AI_CORE_THIRD_PARTY_DIR}/plugins/opencv/include"
            "${AI_CORE_THIRD_PARTY_DIR}/plugins/opencv/modules/core/include"
            "${AI_CORE_THIRD_PARTY_DIR}/plugins/opencv/modules/imgproc/include"
            "${AI_CORE_THIRD_PARTY_DIR}/plugins/opencv/modules/dnn/include"
            "${AI_CORE_THIRD_PARTY_DIR}/plugins/opencv/modules/imgcodecs/include"
            "${CMAKE_BINARY_DIR}")

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
           CMAKE_CXX_FLAGS MATCHES "fsanitize=.*undefined")
            foreach(opencv_target
                    opencv_core opencv_imgproc opencv_dnn opencv_imgcodecs)
                if(TARGET ${opencv_target})
                    target_compile_options(${opencv_target} PRIVATE
                        -fno-sanitize=alignment
                        $<$<CXX_COMPILER_ID:Clang>:-fno-sanitize=function>)
                endif()
            endforeach()
        endif()
    elseif(AI_CORE_OPENCV_PROVIDER STREQUAL "SYSTEM")
        find_package(OpenCV 4 CONFIG REQUIRED COMPONENTS ${opencv_components})
        set(ai_core_opencv_include_dirs ${OpenCV_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR
            "AI_CORE_OPENCV_PROVIDER must be BUNDLED or SYSTEM, got: "
            "${AI_CORE_OPENCV_PROVIDER}")
    endif()

    add_library(ai_core_opencv INTERFACE)
    add_library(ai_core::opencv ALIAS ai_core_opencv)
    target_include_directories(ai_core_opencv SYSTEM INTERFACE
        ${ai_core_opencv_include_dirs})
    target_link_libraries(ai_core_opencv INTERFACE
        opencv_core opencv_imgproc opencv_dnn)

    if("imgcodecs" IN_LIST opencv_components)
        add_library(ai_core_opencv_imgcodecs INTERFACE)
        add_library(ai_core::opencv_imgcodecs ALIAS ai_core_opencv_imgcodecs)
        target_link_libraries(ai_core_opencv_imgcodecs INTERFACE
            ai_core::opencv opencv_imgcodecs)
    endif()

    message(STATUS "OpenCV provider: ${AI_CORE_OPENCV_PROVIDER}")
endfunction()
