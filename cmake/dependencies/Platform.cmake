include_guard(GLOBAL)

function(ai_core_load_platform)
    if(TARGET ai_core::platform)
        return()
    endif()

    add_library(ai_core_platform INTERFACE)
    add_library(ai_core::platform ALIAS ai_core_platform)
    if(ANDROID)
        target_link_libraries(ai_core_platform INTERFACE
            android log z dl mediandk)
    endif()
endfunction()
