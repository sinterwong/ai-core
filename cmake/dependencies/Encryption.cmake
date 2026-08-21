include_guard(GLOBAL)
include("${AI_CORE_CMAKE_DIR}/dependencies/Common.cmake")

function(ai_core_load_encryption)
    if(TARGET encrypt::encrypt)
        return()
    endif()

    ai_core_require_source(
        "encryption-tool" "plugins/encryption_tool" "decryption")
    add_subdirectory(
        "${AI_CORE_THIRD_PARTY_DIR}/plugins/encryption_tool"
        "${CMAKE_BINARY_DIR}/_deps/encryption_tool"
        EXCLUDE_FROM_ALL)
endfunction()
