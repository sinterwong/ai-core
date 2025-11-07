cmake_minimum_required(VERSION 3.18)
project(Benchmarks)

load_opencv()

if(TARGET_OS STREQUAL "Android")
    load_android_env()
    list(APPEND ANDROID_JIN_LIBS mediandk)
endif()

set(DEPENDENCY_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ort
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ncnn
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/trt
)

set(DEPENDENCY_LIBS
    ai_core::ai_core
    logger::logger
    encrypt::encrypt
    benchmark::benchmark
    ${OpenCV_LIBS}
    ${ANDROID_JIN_LIBS}
)

if(WITH_ORT_ENGINE)
    load_onnxruntime()
    list(APPEND DEPENDENCY_INCLUDES
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime
    )
    list(APPEND DEPENDENCY_LIBS ${ONNXRUNTIME_LIBS})
endif()

if(WITH_NCNN_ENGINE)
    load_openmp()
    load_ncnn()

    list(APPEND DEPENDENCY_INCLUDES
        ${NCNN_INCLUDE_DIR}
    )
    list(APPEND DEPENDENCY_LIBS ${NCNN_LIBS} ${OpenMP_CXX_FLAGS})
endif()

if(WITH_TRT_ENGINE)
    load_cuda()
    load_tensorrt()

    list(APPEND DEPENDENCY_INCLUDES
        ${TensorRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/trt
    )

    list(APPEND DEPENDENCY_LIBS TensorRT::nvinfer)
endif()

file(GLOB CURRENT_DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

message(STATUS "Benchmark sources: ${CURRENT_DIR_SRCS}")

file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/benchmarks)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/benchmarks)

add_executable(ai_core_benchmarks ${CURRENT_DIR_SRCS})
target_include_directories(ai_core_benchmarks PRIVATE ${DEPENDENCY_INCLUDES})
target_link_libraries(ai_core_benchmarks PRIVATE ${DEPENDENCY_LIBS})

# Add compile definitions
if(WITH_ORT_ENGINE)
    target_compile_definitions(ai_core_benchmarks PRIVATE WITH_ORT)
endif()

if(WITH_NCNN_ENGINE)
    target_compile_definitions(ai_core_benchmarks PRIVATE WITH_NCNN)
endif()

if(WITH_TRT_ENGINE)
    target_compile_definitions(ai_core_benchmarks PRIVATE WITH_TRT)
endif()

install(TARGETS ai_core_benchmarks DESTINATION benchmarks)
