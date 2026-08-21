cmake_minimum_required(VERSION 3.18)
project(AICore3rdparty)

set(BUILD_TESTING OFF CACHE BOOL "Disable testing for 3rdparty libs")
set(WITH_GTEST OFF CACHE BOOL "Disable gtest for glog")

# OpenCV is source-vendored and deliberately configured here so that all
# dependency policy stays below third_party/. Consumers above this directory only
# request OpenCV targets; they do not control how OpenCV itself is built.
if(AI_CORE_BUILD_PLUGIN_PREPROC_OPENCV OR
   AI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV OR
   AI_CORE_BUILD_PLUGIN_ONNXRUNTIME OR
   AI_CORE_BUILD_PLUGIN_NCNN OR
   AI_CORE_BUILD_PLUGIN_TENSORRT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugins/opencv/CMakeLists.txt")
        set(AI_CORE_OPENCV_MODULES core,imgproc,dnn)
        if(AI_CORE_BUILD_TESTS)
            list(APPEND AI_CORE_OPENCV_MODULES imgcodecs)
        endif()
        list(JOIN AI_CORE_OPENCV_MODULES "," AI_CORE_OPENCV_BUILD_LIST)
        set(BUILD_LIST "${AI_CORE_OPENCV_BUILD_LIST}" CACHE STRING
            "OpenCV modules required by ai-core plugins" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build bundled OpenCV statically" FORCE)
        set(BUILD_TESTS OFF CACHE BOOL "Disable OpenCV tests" FORCE)
        set(BUILD_PERF_TESTS OFF CACHE BOOL "Disable OpenCV performance tests" FORCE)
        set(BUILD_opencv_apps OFF CACHE BOOL "Disable OpenCV applications" FORCE)
        set(BUILD_JAVA OFF CACHE BOOL "Disable OpenCV Java bindings" FORCE)
        set(BUILD_opencv_python3 OFF CACHE BOOL "Disable OpenCV Python bindings" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "Disable OpenCV examples" FORCE)
        # IPP enables an OpenCV configure-time download of a prebuilt IPPICV
        # archive. Keep source checkouts self-contained and reproducible.
        set(WITH_IPP OFF CACHE BOOL "Disable downloaded IPPICV binaries" FORCE)
        set(WITH_ITT OFF CACHE BOOL "Disable Intel ITT instrumentation" FORCE)
        set(WITH_OPENCL OFF CACHE BOOL "Disable OpenCV OpenCL runtime" FORCE)
        set(WITH_OPENGL OFF CACHE BOOL "Disable OpenCV OpenGL support" FORCE)
        set(WITH_VA OFF CACHE BOOL "Disable OpenCV VA-API support" FORCE)
        set(WITH_VA_INTEL OFF CACHE BOOL "Disable Intel VA extensions" FORCE)
        set(WITH_VTK OFF CACHE BOOL "Disable OpenCV VTK support" FORCE)
        set(WITH_GTK OFF CACHE BOOL "Disable OpenCV GTK support" FORCE)

        if(NOT TARGET opencv_core)
            add_subdirectory(plugins/opencv EXCLUDE_FROM_ALL)
        endif()

        # OpenCV's x86 SIMD resize implementation deliberately performs
        # unaligned scalar loads. UBSan's alignment check diagnoses those
        # internals even though the instructions support unaligned access.
        # Keep sanitizers enabled for ai-core while excluding only this check
        # from the source-vendored OpenCV targets.
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
           CMAKE_CXX_FLAGS MATCHES "fsanitize=.*undefined")
            foreach(AI_CORE_OPENCV_TARGET
                    opencv_core opencv_imgproc opencv_dnn opencv_imgcodecs)
                if(TARGET ${AI_CORE_OPENCV_TARGET})
                    target_compile_options(${AI_CORE_OPENCV_TARGET} PRIVATE
                        -fno-sanitize=alignment
                        $<$<CXX_COMPILER_ID:Clang>:-fno-sanitize=function>)
                endif()
            endforeach()
        endif()
    endif()
endif()

if((AI_CORE_BUILD_PLUGIN_ONNXRUNTIME OR
    AI_CORE_BUILD_PLUGIN_NCNN OR
    AI_CORE_BUILD_PLUGIN_TENSORRT) AND NOT TARGET encrypt)
    message(STATUS "encrypt target not found. Adding subdirectory...")
    add_subdirectory(plugins/encryption_tool EXCLUDE_FROM_ALL)
endif()

# libai_core.so and the test binaries need libencrypt.so at runtime, but
# EXCLUDE_FROM_ALL skips encryption_tool's own install rules, so install the
# runtime artifact alongside the project's libraries.
if(TARGET encrypt)
    set_target_properties(encrypt PROPERTIES EXCLUDE_FROM_ALL FALSE)
    install(TARGETS encrypt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(AI_CORE_BUILD_TESTS)
    if(NOT TARGET gtest)
        message(STATUS "gtest target not found. Adding subdirectory...")
        add_subdirectory(testing/googletest EXCLUDE_FROM_ALL)
    else()
        message(STATUS "gtest target already exists. Skipping.")
    endif()
endif()

if(AI_CORE_BUILD_BENCHMARKS)
    set(BUILD_SHARED_LIBS OFF)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable test for benchmark")
    if(NOT TARGET benchmark)
        message(STATUS "benchmark target not found. Adding subdirectory...")
        add_subdirectory(benchmarking/google_benchmark EXCLUDE_FROM_ALL)
    else()
        message(STATUS "benchmark target already exists. Skipping.")
    endif()
endif()

if(AI_CORE_BUILD_CONFIG)
    file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
    file(COPY config/nlohmann_json/include/nlohmann
        DESTINATION ${PROJECT_INCLUDE_DIR})
endif()
