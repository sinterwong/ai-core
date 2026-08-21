#include "ai_core/plugin_manager.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <unordered_set>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ai_core::dnn {
namespace {
namespace fs = std::filesystem;

void closeLibrary(void *handle) {
#if defined(_WIN32)
  FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

bool isSharedLibrary(const fs::path &path) {
  const auto filename = path.filename().string();
  if (filename.find("ai_core_plugin_") == std::string::npos) {
    return false;
  }
#if defined(_WIN32)
  return path.extension() == ".dll";
#elif defined(__APPLE__)
  return filename.ends_with(".dylib") ||
         filename.find(".so") != std::string::npos;
#else
  return filename.find(".so") != std::string::npos;
#endif
}

std::vector<fs::path> splitSearchPath(const char *value) {
  std::vector<fs::path> result;
  if (!value) {
    return result;
  }
#if defined(_WIN32)
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  std::string paths(value);
  size_t begin = 0;
  while (begin <= paths.size()) {
    const auto end = paths.find(separator, begin);
    const auto item = paths.substr(begin, end - begin);
    if (!item.empty()) {
      result.emplace_back(item);
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return result;
}
} // namespace

PluginManager &PluginManager::instance() {
  static PluginManager manager;
  return manager;
}

PluginInfo PluginManager::load(const std::string &path) {
  std::lock_guard lock(m_mutex);
  std::error_code path_error;
  const auto canonical_path =
      fs::weakly_canonical(fs::path(path), path_error).string();
  const auto &identity = path_error ? path : canonical_path;
  for (size_t i = 0; i < m_paths.size(); ++i) {
    if (m_paths[i] == identity) {
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
    closeLibrary(reinterpret_cast<void *>(handle));
    throw std::runtime_error("Library '" + path + "' does not export " +
                             std::string(AI_CORE_PLUGIN_ENTRYPOINT));
  }

  PluginInfo info;
  auto &registry = PluginRegistry::instance();
  auto snapshot = registry.snapshot();
  bool registered = false;
  try {
    registered = entry(registry, info);
  } catch (...) {
    registry.restore(std::move(snapshot));
    closeLibrary(reinterpret_cast<void *>(handle));
    throw;
  }
  if (!registered || info.name.empty() ||
      info.api_version != AI_CORE_PLUGIN_API_VERSION) {
    registry.restore(std::move(snapshot));
    closeLibrary(reinterpret_cast<void *>(handle));
    throw std::runtime_error("Plugin registration failed for '" + path + "'");
  }
  m_handles.push_back(reinterpret_cast<void *>(handle));
  m_paths.push_back(identity);
  m_plugins.push_back(info);
  return info;
}

std::vector<PluginInfo>
PluginManager::loadDirectory(const std::filesystem::path &directory) {
  std::vector<fs::path> libraries;
  std::unordered_set<std::string> identities;
  std::error_code error;
  if (!fs::is_directory(directory, error)) {
    return {};
  }
  for (fs::directory_iterator it(directory, error), end; !error && it != end;
       it.increment(error)) {
    if (it->is_regular_file() && isSharedLibrary(it->path())) {
      auto identity = fs::weakly_canonical(it->path(), error).string();
      if (!error && identities.insert(identity).second) {
        libraries.push_back(it->path());
      }
      error.clear();
    }
  }
  std::sort(libraries.begin(), libraries.end());
  std::vector<PluginInfo> result;
  result.reserve(libraries.size());
  for (const auto &library : libraries) {
    result.push_back(load(library.string()));
  }
  return result;
}

std::vector<std::filesystem::path> PluginManager::defaultSearchPaths() const {
  auto paths = splitSearchPath(std::getenv("AI_CORE_PLUGIN_PATH"));
#if !defined(_WIN32)
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void *>(&PluginManager::instance), &info) &&
      info.dli_fname) {
    paths.push_back(fs::path(info.dli_fname).parent_path() / "ai_core" /
                    "plugins");
  }
#endif
  return paths;
}

std::vector<PluginInfo> PluginManager::discover(
    const std::vector<std::filesystem::path> &search_paths) {
  auto paths = search_paths;
  const auto defaults = defaultSearchPaths();
  paths.insert(paths.end(), defaults.begin(), defaults.end());
  std::vector<PluginInfo> result;
  for (const auto &path : paths) {
    auto loaded = loadDirectory(path);
    result.insert(result.end(), loaded.begin(), loaded.end());
  }
  return result;
}

std::vector<PluginInfo> PluginManager::loadedPlugins() const {
  std::lock_guard lock(m_mutex);
  return m_plugins;
}

} // namespace ai_core::dnn
