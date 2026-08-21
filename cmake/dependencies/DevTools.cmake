include_guard(GLOBAL)
include("${AI_CORE_CMAKE_DIR}/dependencies/Common.cmake")

function(ai_core_load_googletest)
    if(TARGET GTest::gtest)
        return()
    endif()
    ai_core_require_source("GoogleTest" "testing/googletest" "testing")
    set(BUILD_GMOCK OFF CACHE BOOL "Disable GoogleMock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest install" FORCE)
    add_subdirectory(
        "${AI_CORE_THIRD_PARTY_DIR}/testing/googletest"
        "${CMAKE_BINARY_DIR}/_deps/googletest"
        EXCLUDE_FROM_ALL)
endfunction()

function(ai_core_load_benchmark)
    if(TARGET benchmark::benchmark)
        return()
    endif()
    ai_core_require_source(
        "Google Benchmark" "benchmarking/google_benchmark" "benchmarking")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
        "Disable Google Benchmark tests" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL
        "Disable Google Benchmark install" FORCE)
    add_subdirectory(
        "${AI_CORE_THIRD_PARTY_DIR}/benchmarking/google_benchmark"
        "${CMAKE_BINARY_DIR}/_deps/google_benchmark"
        EXCLUDE_FROM_ALL)
endfunction()
