include_guard(GLOBAL)
include("${AI_CORE_CMAKE_DIR}/dependencies/Common.cmake")

function(ai_core_load_nlohmann_json)
    if(TARGET nlohmann_json::nlohmann_json)
        return()
    endif()

    ai_core_require_source(
        "nlohmann/json" "config/nlohmann_json" "config")
    set(JSON_BuildTests OFF CACHE BOOL "Disable nlohmann/json tests" FORCE)
    add_subdirectory(
        "${AI_CORE_THIRD_PARTY_DIR}/config/nlohmann_json"
        "${CMAKE_BINARY_DIR}/_deps/nlohmann_json"
        EXCLUDE_FROM_ALL)
endfunction()
