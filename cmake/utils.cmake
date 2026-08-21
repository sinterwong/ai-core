function(ai_core_extract_version)
    set(target_ver_file
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ai_core/version.hpp")

    if(NOT EXISTS "${target_ver_file}")
        message(FATAL_ERROR
            "Could not find public version header: ${target_ver_file}")
    endif()

    file(READ "${target_ver_file}" file_contents)
    
    string(REGEX MATCH "AI_CORE_VER_MAJOR ([0-9]+)" _ "${file_contents}")
    if(NOT CMAKE_MATCH_COUNT EQUAL 1)
        message(FATAL_ERROR "Could not extract major version number from version.hpp")
    endif()
    set(ver_major ${CMAKE_MATCH_1})

    string(REGEX MATCH "AI_CORE_VER_MINOR ([0-9]+)" _ "${file_contents}")
    if(NOT CMAKE_MATCH_COUNT EQUAL 1)
        message(FATAL_ERROR "Could not extract minor version number from version.hpp")
    endif()
    set(ver_minor ${CMAKE_MATCH_1})

    string(REGEX MATCH "AI_CORE_VER_PATCH ([0-9]+)" _ "${file_contents}")
    if(NOT CMAKE_MATCH_COUNT EQUAL 1)
        message(FATAL_ERROR "Could not extract patch version number from version.hpp")
    endif()
    set(ver_patch ${CMAKE_MATCH_1})

    set(AI_CORE_VERSION_MAJOR ${ver_major} PARENT_SCOPE)
    set(AI_CORE_VERSION "${ver_major}.${ver_minor}.${ver_patch}" PARENT_SCOPE)
endfunction()
