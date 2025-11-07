cmake_minimum_required(VERSION 3.18)
project(Resources)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION ./assets)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conf")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/conf" DESTINATION ./assets)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/enc_models")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/enc_models" DESTINATION ./assets)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/models" DESTINATION ./assets)
endif()
