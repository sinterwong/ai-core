
#pragma once

// for cmake
#define AI_CORE_VER_MAJOR 2
#define AI_CORE_VER_MINOR 0
#define AI_CORE_VER_PATCH 0

#define AI_CORE_VERSION                                                        \
  (AI_CORE_VER_MAJOR * 10000 + AI_CORE_VER_MINOR * 100 + AI_CORE_VER_PATCH)

// for source code
#define _AI_CORE_STR(s) #s
#define AI_CORE_PROJECT_VERSION(major, minor, patch)                           \
  "v" _AI_CORE_STR(major.minor.patch)

#define AI_CORE_VERSION_STR                                                    \
  AI_CORE_PROJECT_VERSION(AI_CORE_VER_MAJOR, AI_CORE_VER_MINOR,                \
                          AI_CORE_VER_PATCH)
