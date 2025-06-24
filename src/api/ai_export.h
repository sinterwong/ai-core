#ifndef __AI_CORE_SDK_EXPORT_H_
#define __AI_CORE_SDK_EXPORT_H_

#ifdef _WIN32
#ifdef AI_CORE_SDK_BUILD_DLL
#define AI_CORE_SDK_API __declspec(dllexport)
#else
#define AI_CORE_SDK_API __declspec(dllimport)
#endif
#else
#define AI_CORE_SDK_API __attribute__((visibility("default")))
#endif

#endif // __AI_CORE_SDK_EXPORT_H_
