CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(AICore3rdparty)

SET(BUILD_TESTS OFF CACHE BOOL "Disable tests")
SET(BUILD_TOOLS OFF CACHE BOOL "Disable tools")

ADD_SUBDIRECTORY(cpp-common-utils EXCLUDE_FROM_ALL)

ADD_SUBDIRECTORY(encryption-tool EXCLUDE_FROM_ALL)

IF(NOT TARGET logger)
    MESSAGE(STATUS "logger target not found. Adding subdirectory...")
    ADD_SUBDIRECTORY(logger EXCLUDE_FROM_ALL)
ELSE()
    MESSAGE(STATUS "logger target already exists. Skipping.")
ENDIF()

IF(NOT TARGET gtest)
    MESSAGE(STATUS "gtest target not found. Adding subdirectory...")
    ADD_SUBDIRECTORY(googletest EXCLUDE_FROM_ALL)
ELSE()
    MESSAGE(STATUS "gtest target already exists. Skipping.")
ENDIF()
