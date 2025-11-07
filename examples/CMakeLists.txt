cmake_minimum_required(VERSION 3.18)

project(AICoreExamples VERSION ${AI_CORE_VERSION})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(COMMIT_CODE "689bc3e3bdf1c5f2cff81725011ba7d3c0089b25")

load_opencv()

if(TARGET_OS STREQUAL "Android")
    load_android_env()
    list(APPEND ANDROID_JIN_LIBS mediandk)
endif()

file(GLOB CURRENT_DIR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

set(DEPENDENCY_INCLUDES
    ${PROJECT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(DEPENDENCY_LIBS
    ai_core::ai_core
    logger::logger
    ${OpenCV_LIBS}
    ${ANDROID_JIN_LIBS}
)

add_library(ai_core_examples STATIC ${CURRENT_DIR_SRCS})

add_library(ai_core::ai_core_examples ALIAS ai_core_examples)

target_include_directories(ai_core_examples
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        ${DEPENDENCY_INCLUDES}
)

target_link_libraries(ai_core_examples PRIVATE ${DEPENDENCY_LIBS})

target_compile_definitions(ai_core_examples PRIVATE SECURITY_KEY="${COMMIT_CODE}")

install(TARGETS ai_core_examples
    EXPORT ai_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)

add_subdirectory(ocr)
