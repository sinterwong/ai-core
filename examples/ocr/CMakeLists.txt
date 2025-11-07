cmake_minimum_required(VERSION 3.18)

project(AICoreExamplesOCRReco VERSION ${AI_CORE_VERSION})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

load_opencv()

file(GLOB_RECURSE CURRENT_DIR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

set(DEPENDENCY_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(DEPENDENCY_LIBS
    ai_core::ai_core_examples
    ai_core::ai_core
    logger::logger
    ${OpenCV_LIBS}
)

add_executable(ai_core_example_ocr ${CURRENT_DIR_SRCS})

target_include_directories(ai_core_example_ocr PRIVATE ${DEPENDENCY_INCLUDES})

target_link_libraries(ai_core_example_ocr PRIVATE ${DEPENDENCY_LIBS})

install(TARGETS ai_core_example_ocr
    EXPORT ai_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)
