#include "ai_core/plugin_manager.hpp"

#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ai_core::dnn {

PluginManager &PluginManager::instance() {
  static PluginManager manager;
  return manager;
}

PluginInfo PluginManager::load(const std::string &path) {
  std::lock_guard lock(m_mutex);
  for (size_t i = 0; i < m_paths.size(); ++i) {
    if (m_paths[i] == path) {
      return m_plugins[i];
    }
  }

#if defined(_WIN32)
  HMODULE handle = LoadLibraryA(path.c_str());
  if (!handle) {
    throw std::runtime_error("Failed to load ai-core plugin: " + path);
  }
  auto entry = reinterpret_cast<AiCoreRegisterPluginV1>(
      GetProcAddress(handle, AI_CORE_PLUGIN_ENTRYPOINT.data()));
#else
  dlerror();
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char *error = dlerror();
    throw std::runtime_error("Failed to load ai-core plugin '" + path +
                             "': " + (error ? error : "unknown error"));
  }
  auto entry = reinterpret_cast<AiCoreRegisterPluginV1>(
      dlsym(handle, AI_CORE_PLUGIN_ENTRYPOINT.data()));
#endif
  if (!entry) {
    throw std::runtime_error("Library '" + path + "' does not export " +
                             std::string(AI_CORE_PLUGIN_ENTRYPOINT));
  }

  PluginInfo info;
  if (!entry(PluginRegistry::instance(), info) || info.name.empty()) {
    throw std::runtime_error("Plugin registration failed for '" + path + "'");
  }
  m_handles.push_back(reinterpret_cast<void *>(handle));
  m_paths.push_back(path);
  m_plugins.push_back(info);
  return info;
}

std::vector<PluginInfo> PluginManager::loadedPlugins() const {
  std::lock_guard lock(m_mutex);
  return m_plugins;
}

} // namespace ai_core::dnn
