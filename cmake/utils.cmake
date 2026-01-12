function(ai_core_extract_version)
    set(possible_paths 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/api/ai_core/ai_core_version.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/api/ai_core/ai_core_version.hpp"
    )
    
    set(target_ver_file "")
    foreach(p ${possible_paths})
        if(EXISTS "${p}")
            set(target_ver_file "${p}")
            break()
        endif()
    endforeach()

    if(NOT target_ver_file)
        message(FATAL_ERROR "Could not find ai_core_version.hpp in src/api or api/")
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
