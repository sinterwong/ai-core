include_guard(GLOBAL)

function(ai_core_configure_code_quality)
    if(AI_CORE_CLANG_FORMAT_EXECUTABLE AND
       (NOT IS_ABSOLUTE "${AI_CORE_CLANG_FORMAT_EXECUTABLE}" OR
        NOT EXISTS "${AI_CORE_CLANG_FORMAT_EXECUTABLE}"))
        get_filename_component(clang_format_name
            "${AI_CORE_CLANG_FORMAT_EXECUTABLE}" NAME)
        unset(AI_CORE_CLANG_FORMAT_RESOLVED CACHE)
        find_program(AI_CORE_CLANG_FORMAT_RESOLVED
            NAMES "${clang_format_name}")
        if(AI_CORE_CLANG_FORMAT_RESOLVED)
            set(AI_CORE_CLANG_FORMAT_EXECUTABLE
                "${AI_CORE_CLANG_FORMAT_RESOLVED}"
                CACHE FILEPATH "clang-format 21 executable used by ai-core"
                FORCE)
        endif()
    else()
        find_program(AI_CORE_CLANG_FORMAT_EXECUTABLE
            NAMES clang-format-21 clang-format
            DOC "clang-format 21 executable used by ai-core")
    endif()
    unset(AI_CORE_CLANG_FORMAT_RESOLVED CACHE)
    mark_as_advanced(AI_CORE_CLANG_FORMAT_EXECUTABLE)

    if(NOT AI_CORE_CLANG_FORMAT_EXECUTABLE)
        if(AI_CORE_FORMAT_CHECK_ON_BUILD)
            message(WARNING
                "clang-format 21 was not found; the on-build format check is "
                "disabled. Install clang-format-21 or set "
                "AI_CORE_CLANG_FORMAT_EXECUTABLE.")
        endif()
        return()
    endif()

    execute_process(
        COMMAND "${AI_CORE_CLANG_FORMAT_EXECUTABLE}" --version
        OUTPUT_VARIABLE clang_format_version
        ERROR_VARIABLE clang_format_error
        RESULT_VARIABLE clang_format_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT clang_format_result EQUAL 0)
        if(AI_CORE_FORMAT_CHECK_ON_BUILD)
            message(WARNING
                "Could not run ${AI_CORE_CLANG_FORMAT_EXECUTABLE}: "
                "${clang_format_error}. Format targets are disabled.")
        endif()
        return()
    endif()
    if(NOT clang_format_version MATCHES "clang-format version 21\\.")
        if(AI_CORE_FORMAT_CHECK_ON_BUILD)
            message(WARNING
                "ai-core formatting is pinned to clang-format 21, but found: "
                "${clang_format_version}. Format targets are disabled.")
        endif()
        return()
    endif()

    set(format_sources)
    foreach(source_root IN ITEMS
            benchmarks
            examples
            include
            plugins
            src
            tests)
        foreach(extension IN ITEMS hpp cpp cc cu)
            file(GLOB_RECURSE matching_sources CONFIGURE_DEPENDS
                "${AI_CORE_SOURCE_DIR}/${source_root}/*.${extension}")
            list(APPEND format_sources ${matching_sources})
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES format_sources)
    list(SORT format_sources)

    if(AI_CORE_FORMAT_CHECK_ON_BUILD)
        set(format_check_all ALL)
    else()
        set(format_check_all)
    endif()

    add_custom_target(ai_core_format_check ${format_check_all}
        COMMAND "${AI_CORE_CLANG_FORMAT_EXECUTABLE}"
            --dry-run --Werror ${format_sources}
        WORKING_DIRECTORY "${AI_CORE_SOURCE_DIR}"
        COMMENT "Checking ai-core formatting with clang-format 21"
        VERBATIM)

    add_custom_target(ai_core_format
        COMMAND "${AI_CORE_CLANG_FORMAT_EXECUTABLE}" -i ${format_sources}
        WORKING_DIRECTORY "${AI_CORE_SOURCE_DIR}"
        COMMENT "Formatting ai-core sources with clang-format 21"
        VERBATIM)
endfunction()
