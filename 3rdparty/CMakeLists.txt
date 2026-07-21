cmake_minimum_required(VERSION 3.18)
project(AICore3rdparty)

set(BUILD_TESTING OFF CACHE BOOL "Disable testing for 3rdparty libs")
set(WITH_GTEST OFF CACHE BOOL "Disable gtest for glog")

if(NOT TARGET encrypt)
    message(STATUS "encrypt target not found. Adding subdirectory...")
    add_subdirectory(encryption-tool EXCLUDE_FROM_ALL)
endif()

# libai_core.so and the test binaries need libencrypt.so at runtime, but
# EXCLUDE_FROM_ALL skips encryption-tool's own install rules, so install the
# runtime artifact alongside the project's libraries.
set_target_properties(encrypt PROPERTIES EXCLUDE_FROM_ALL FALSE)
install(TARGETS encrypt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

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
