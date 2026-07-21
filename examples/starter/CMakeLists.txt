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

# ai_core-config.cmake pulls in its own dependencies (OpenCV) via
# find_dependency, so this single call is enough.
find_package(ai_core CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ai_core_starter main.cpp)
target_link_libraries(ai_core_starter
    PRIVATE
        ai_core::ai_core          # core inference API
        ai_core::ai_core_config   # optional JSON config module
        ${OpenCV_LIBS}            # only for image I/O in this example
)
target_include_directories(ai_core_starter PRIVATE ${OpenCV_INCLUDE_DIRS})
