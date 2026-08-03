#include "ai_core/plugin_registry.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ai_core::dnn {
namespace {
template <class Map>
std::vector<std::string> namesOf(const Map &map) {
  std::vector<std::string> result;
  result.reserve(map.size());
  for (const auto &[name, unused] : map) {
    (void)unused;
    result.push_back(name);
  }
  std::sort(result.begin(), result.end());
  return result;
}

template <class Map>
bool contains(const Map &map, std::string_view name) {
  return map.find(std::string(name)) != map.end();
}

template <class Map>
auto create(const Map &map, std::string_view name, const DataPacket &params) {
  auto it = map.find(std::string(name));
  if (it == map.end()) {
    throw std::runtime_error("Plugin '" + std::string(name) +
                             "' is not registered");
  }
  return it->second(params);
}
} // namespace

PluginRegistry &PluginRegistry::instance() {
  static PluginRegistry registry;
  return registry;
}

template <class Map, class Creator>
bool PluginRegistry::add(Map &map, std::string name, Creator creator) {
  if (name.empty() || !creator) {
    throw std::invalid_argument("Plugin name and creator must not be empty");
  }
  std::unique_lock lock(m_mutex);
  return map.emplace(std::move(name), std::move(creator)).second;
}

bool PluginRegistry::registerPreprocessor(std::string name,
                                          PreprocCreator creator) {
  return add(m_preprocessors, std::move(name), std::move(creator));
}
bool PluginRegistry::registerInferenceEngine(std::string name,
                                             InferCreator creator) {
  return add(m_inferenceEngines, std::move(name), std::move(creator));
}
bool PluginRegistry::registerPostprocessor(std::string name,
                                           PostprocCreator creator) {
  return add(m_postprocessors, std::move(name), std::move(creator));
}

std::shared_ptr<IPreprocessPlugin>
PluginRegistry::createPreprocessor(std::string_view name,
                                   const DataPacket &params) const {
  std::shared_lock lock(m_mutex);
  return create(m_preprocessors, name, params);
}
std::shared_ptr<IInferEnginePlugin>
PluginRegistry::createInferenceEngine(std::string_view name,
                                      const DataPacket &params) const {
  std::shared_lock lock(m_mutex);
  return create(m_inferenceEngines, name, params);
}
std::shared_ptr<IPostprocessPlugin>
PluginRegistry::createPostprocessor(std::string_view name,
                                    const DataPacket &params) const {
  std::shared_lock lock(m_mutex);
  return create(m_postprocessors, name, params);
}

bool PluginRegistry::containsPreprocessor(std::string_view name) const {
  std::shared_lock lock(m_mutex);
  return contains(m_preprocessors, name);
}
bool PluginRegistry::containsInferenceEngine(std::string_view name) const {
  std::shared_lock lock(m_mutex);
  return contains(m_inferenceEngines, name);
}
bool PluginRegistry::containsPostprocessor(std::string_view name) const {
  std::shared_lock lock(m_mutex);
  return contains(m_postprocessors, name);
}
std::vector<std::string> PluginRegistry::preprocessors() const {
  std::shared_lock lock(m_mutex);
  return namesOf(m_preprocessors);
}
std::vector<std::string> PluginRegistry::inferenceEngines() const {
  std::shared_lock lock(m_mutex);
  return namesOf(m_inferenceEngines);
}
std::vector<std::string> PluginRegistry::postprocessors() const {
  std::shared_lock lock(m_mutex);
  return namesOf(m_postprocessors);
}

} // namespace ai_core::dnn
