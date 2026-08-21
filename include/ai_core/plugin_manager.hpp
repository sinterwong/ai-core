#ifndef AI_CORE_PLUGIN_MANAGER_HPP
#define AI_CORE_PLUGIN_MANAGER_HPP

#include "ai_core/plugin_registry.hpp"

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ai_core::dnn {

inline constexpr unsigned AI_CORE_PLUGIN_API_VERSION = 1;
inline constexpr std::string_view AI_CORE_PLUGIN_ENTRYPOINT =
    "ai_core_register_plugin_v1";

/** Metadata filled by a plugin's versioned registration entrypoint. */
struct PluginInfo {
  std::uint32_t api_version{AI_CORE_PLUGIN_API_VERSION};
  std::string name;
  std::string version;
  std::string provider;
  std::string description;
  std::vector<std::string> capabilities;
};

/**
 * @brief Loads plugin shared libraries and keeps them resident for the process.
 *
 * A load is transactional: registrations are rolled back if the entrypoint is
 * absent, rejects the API version, or throws.
 *
 * @par Thread safety
 * All methods may be called concurrently; loads and queries are serialized
 * internally.
 */
class PluginManager final {
public:
  static PluginManager &instance();

  /**
   * @brief Load one shared library and invoke its registration entrypoint.
   * @throws std::runtime_error If loading or registration fails.
   */
  PluginInfo load(const std::string &path);

  /** Load every candidate plugin directly inside `directory`. */
  std::vector<PluginInfo> loadDirectory(const std::filesystem::path &directory);

  /** Search explicit paths, or the platform defaults when none are supplied. */
  std::vector<PluginInfo>
  discover(const std::vector<std::filesystem::path> &search_paths = {});
  std::vector<std::filesystem::path> defaultSearchPaths() const;
  /** Return a snapshot of successfully loaded plugin metadata. */
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
using AiCoreRegisterPluginV1 = bool (*)(ai_core::dnn::PluginRegistry &,
                                        ai_core::dnn::PluginInfo &);
}

#endif
