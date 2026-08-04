#pragma once

#include <filesystem>

inline std::filesystem::path aiCoreTestPluginPath(const char *argv0,
                                                  const char *build_path) {
  std::error_code error;
  const auto executable = std::filesystem::weakly_canonical(argv0, error);
  if (!error) {
    const auto candidate = executable.parent_path().parent_path() / "lib" /
                           "ai_core" / "plugins" /
                           std::filesystem::path(build_path).filename();
    if (std::filesystem::exists(candidate, error) && !error) {
      return candidate;
    }
  }
  return build_path;
}
