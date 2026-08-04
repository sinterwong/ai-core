include(CMakeParseArguments)

function(ai_core_add_plugin target)
    set(one_value_args TYPE)
    set(multi_value_args SOURCES DEPENDENCIES)
    cmake_parse_arguments(AICP "" "${one_value_args}"
                          "${multi_value_args}" ${ARGN})
    if(NOT AICP_TYPE MATCHES "^(preproc|infer|postproc|bundle)$")
        message(FATAL_ERROR
            "ai_core_add_plugin(${target}): invalid or missing TYPE")
    endif()
    if(NOT AICP_SOURCES)
        message(FATAL_ERROR "ai_core_add_plugin(${target}): SOURCES is required")
    endif()
    add_library(${target} SHARED ${AICP_SOURCES})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE ai_core::ai_core
                          ${AICP_DEPENDENCIES})
endfunction()
