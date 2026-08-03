#ifndef AI_CORE_PLUGIN_MANAGER_HPP
#define AI_CORE_PLUGIN_MANAGER_HPP

#include "ai_core/plugin_registry.hpp"

#include <mutex>
#include <filesystem>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ai_core::dnn {

inline constexpr unsigned AI_CORE_PLUGIN_API_VERSION = 1;
inline constexpr std::string_view AI_CORE_PLUGIN_ENTRYPOINT =
    "ai_core_register_plugin_v1";

struct PluginInfo {
  std::uint32_t api_version{AI_CORE_PLUGIN_API_VERSION};
  std::string name;
  std::string version;
  std::string provider;
  std::string description;
  std::vector<std::string> capabilities;
};

/** Loads plugin DSOs and keeps them resident for the process lifetime. */
class PluginManager final {
public:
  static PluginManager &instance();

  PluginInfo load(const std::string &path);
  std::vector<PluginInfo> loadDirectory(const std::filesystem::path &directory);
  std::vector<PluginInfo>
  discover(const std::vector<std::filesystem::path> &search_paths = {});
  std::vector<std::filesystem::path> defaultSearchPaths() const;
  std::vector<PluginInfo> loadedPlugins() const;

private:
  PluginManager() = default;

  mutable std::mutex m_mutex;
  std::vector<void *> m_handles;
  std::vector<std::string> m_paths;
  std::vector<PluginInfo> m_plugins;
};

} // namespace ai_core::dnn

#if defined(_WIN32)
#define AI_CORE_PLUGIN_EXPORT __declspec(dllexport)
#else
#define AI_CORE_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
using AiCoreRegisterPluginV1 = bool (*)(
    ai_core::dnn::PluginRegistry &, ai_core::dnn::PluginInfo &);
}

#endif
