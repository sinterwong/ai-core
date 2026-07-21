cmake_minimum_required(VERSION 3.18)

project(AICoreExampleAsyncPipeline VERSION ${AI_CORE_VERSION})

# The async pipeline showcase needs the TensorRT backend (IAsyncInferEngine).
if(NOT WITH_TRT_ENGINE)
    message(STATUS "async_pipeline example skipped (WITH_TRT_ENGINE=OFF)")
    return()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

set(DEPENDENCY_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_INCLUDE_DIR}
)

add_executable(ai_core_example_async_pipeline async_pipeline_main.cpp)
target_compile_features(ai_core_example_async_pipeline PRIVATE cxx_std_20)
target_include_directories(ai_core_example_async_pipeline
    PRIVATE ${DEPENDENCY_INCLUDES})
target_link_libraries(ai_core_example_async_pipeline
    PRIVATE ai_core::ai_core Threads::Threads)

install(
    TARGETS ai_core_example_async_pipeline
    EXPORT ai_export_set
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)
