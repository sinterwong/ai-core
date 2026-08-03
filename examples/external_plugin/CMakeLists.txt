cmake_minimum_required(VERSION 3.18)
project(ai_core_external_plugin_example LANGUAGES CXX)

find_package(ai_core REQUIRED)

ai_core_add_plugin(example_amd_infer
    TYPE infer
    SOURCES amd_infer_plugin.cpp
)
