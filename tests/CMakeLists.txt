cmake_minimum_required(VERSION 3.18)
project(Tests)

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
    ${OpenCV_LIBS}
    GTest::gtest
    GTest::gtest_main
    ${ANDROID_JIN_LIBS}
)

if(WITH_ORT_ENGINE)
    load_onnxruntime()
    list(APPEND DEPENDENCY_LIBS onnxruntime::onnxruntime)
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

message(STATUS "Test sources: ${CURRENT_DIR_SRCS}")

file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/tests)

add_executable(ai_core_tests ${CURRENT_DIR_SRCS})
target_include_directories(ai_core_tests PRIVATE ${DEPENDENCY_INCLUDES})
target_link_libraries(ai_core_tests PRIVATE ${DEPENDENCY_LIBS})

# Add compile definitions
if(WITH_ORT_ENGINE)
    target_compile_definitions(ai_core_tests PRIVATE WITH_ORT)
endif()

if(WITH_NCNN_ENGINE)
    target_compile_definitions(ai_core_tests PRIVATE WITH_NCNN)
endif()

if(WITH_TRT_ENGINE)
    target_compile_definitions(ai_core_tests PRIVATE WITH_TRT)
endif()

install(TARGETS ai_core_tests DESTINATION tests)
