# Standalone starter project: consumes an *installed* ai_core via
# find_package(). This is NOT part of the main build — copy this directory out,
# point CMAKE_PREFIX_PATH at your ai_core install, and build.
#
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/ai_core/install
#   cmake --build build
#   ./build/ai_core_starter conf/model.json data/image.png
cmake_minimum_required(VERSION 3.18)
project(ai_core_starter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ai_core 2.0 CONFIG REQUIRED COMPONENTS
    core
    config
    plugin_preproc_opencv
    plugin_postproc_opencv
    plugin_onnxruntime)
find_package(OpenCV REQUIRED)

add_executable(ai_core_starter main.cpp)
target_link_libraries(ai_core_starter
    PRIVATE
        ai_core::ai_core          # core inference API
        ai_core::config           # optional JSON config module
        ${OpenCV_LIBS}            # only for image I/O in this example
)
target_include_directories(ai_core_starter PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(ai_core_starter PRIVATE
    "AI_CORE_PLUGIN_DIR=\"${ai_core_PLUGIN_DIR}\"")
