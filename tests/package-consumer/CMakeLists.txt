cmake_minimum_required(VERSION 3.18)

project(AICorePackageConsumer LANGUAGES CXX)

find_package(ai_core 2.0 REQUIRED COMPONENTS core)
add_executable(ai_core_package_consumer main.cpp)
target_link_libraries(ai_core_package_consumer PRIVATE ai_core::ai_core)
