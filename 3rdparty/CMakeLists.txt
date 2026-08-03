cmake_minimum_required(VERSION 3.18)
project(AICore3rdparty)

set(BUILD_TESTING OFF CACHE BOOL "Disable testing for 3rdparty libs")
set(WITH_GTEST OFF CACHE BOOL "Disable gtest for glog")

# OpenCV is source-vendored and deliberately configured here so that all
# dependency policy stays below 3rdparty/. Consumers above this directory only
# request OpenCV targets; they do not control how OpenCV itself is built.
if(AI_CORE_BUILD_BUNDLED_PLUGINS AND
   (AI_CORE_PLUGIN_VISION OR WITH_ORT_ENGINE OR WITH_NCNN_ENGINE OR WITH_TRT_ENGINE))
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencv/CMakeLists.txt")
        set(AI_CORE_OPENCV_MODULES core,imgproc,dnn)
        if(BUILD_AI_CORE_TESTS)
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
            add_subdirectory(opencv EXCLUDE_FROM_ALL)
        endif()
    endif()
endif()

if((WITH_ORT_ENGINE OR WITH_NCNN_ENGINE OR WITH_TRT_ENGINE) AND NOT TARGET encrypt)
    message(STATUS "encrypt target not found. Adding subdirectory...")
    add_subdirectory(encryption-tool EXCLUDE_FROM_ALL)
endif()

# libai_core.so and the test binaries need libencrypt.so at runtime, but
# EXCLUDE_FROM_ALL skips encryption-tool's own install rules, so install the
# runtime artifact alongside the project's libraries.
if(TARGET encrypt)
    set_target_properties(encrypt PROPERTIES EXCLUDE_FROM_ALL FALSE)
    install(TARGETS encrypt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(BUILD_AI_CORE_TESTS)
    if(NOT TARGET gtest)
        message(STATUS "gtest target not found. Adding subdirectory...")
        add_subdirectory(googletest EXCLUDE_FROM_ALL)
    else()
        message(STATUS "gtest target already exists. Skipping.")
    endif()
endif()

if(BUILD_AI_CORE_BENCHMARKS)
    set(BUILD_SHARED_LIBS OFF)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable test for benchmark")
    if(NOT TARGET benchmark)
        message(STATUS "benchmark target not found. Adding subdirectory...")
        add_subdirectory(benchmark EXCLUDE_FROM_ALL)
    else()
        message(STATUS "benchmark target already exists. Skipping.")
    endif()
endif()

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(COPY nlohmann DESTINATION ${PROJECT_INCLUDE_DIR})
