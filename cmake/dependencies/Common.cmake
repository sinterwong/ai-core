include_guard(GLOBAL)

set(AI_CORE_THIRD_PARTY_DIR "${AI_CORE_SOURCE_DIR}/third_party")
set(AI_CORE_DEPS_ROOT
    "${AI_CORE_SOURCE_DIR}/.deps/${TARGET_OS}_${TARGET_ARCH}"
    CACHE PATH "Root containing prebuilt SDKs for the target platform")

function(ai_core_require_source dependency relative_path profile)
    set(source_dir "${AI_CORE_THIRD_PARTY_DIR}/${relative_path}")
    if(NOT EXISTS "${source_dir}/CMakeLists.txt")
        message(FATAL_ERROR
            "${dependency} source is not initialized at ${source_dir}.\n"
            "Run: scripts/deps.sh init ${profile}")
    endif()
endfunction()

function(ai_core_require_directory dependency path profile)
    if(NOT IS_DIRECTORY "${path}")
        message(FATAL_ERROR
            "${dependency} SDK was not found at ${path}.\n"
            "Run: scripts/deps.sh init ${profile}\n"
            "or set the corresponding AI_CORE_*_ROOT cache variable.")
    endif()
endfunction()
