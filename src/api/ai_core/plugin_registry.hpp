#ifndef AI_CORE_PLUGIN_REGISTRY_HPP
#define AI_CORE_PLUGIN_REGISTRY_HPP

#include "ai_core/data_packet.hpp"
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/i_preprocess.hpp"

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ai_core::dnn {

class PluginManager;

/** The one process-wide registry used by bundled and out-of-tree plugins. */
class PluginRegistry final {
public:
  using PreprocCreator =
      std::function<std::shared_ptr<IPreprocessPlugin>(const DataPacket &)>;
  using InferCreator =
      std::function<std::shared_ptr<IInferEnginePlugin>(const DataPacket &)>;
  using PostprocCreator =
      std::function<std::shared_ptr<IPostprocessPlugin>(const DataPacket &)>;

  static PluginRegistry &instance();

  bool registerPreprocessor(std::string name, PreprocCreator creator);
  bool registerInferenceEngine(std::string name, InferCreator creator);
  bool registerPostprocessor(std::string name, PostprocCreator creator);

  std::shared_ptr<IPreprocessPlugin>
  createPreprocessor(std::string_view name,
                     const DataPacket &params = {}) const;
  std::shared_ptr<IInferEnginePlugin>
  createInferenceEngine(std::string_view name,
                        const DataPacket &params = {}) const;
  std::shared_ptr<IPostprocessPlugin>
  createPostprocessor(std::string_view name,
                      const DataPacket &params = {}) const;

  bool containsPreprocessor(std::string_view name) const;
  bool containsInferenceEngine(std::string_view name) const;
  bool containsPostprocessor(std::string_view name) const;

  std::vector<std::string> preprocessors() const;
  std::vector<std::string> inferenceEngines() const;
  std::vector<std::string> postprocessors() const;

private:
  struct Snapshot {
    std::unordered_map<std::string, PreprocCreator> preprocessors;
    std::unordered_map<std::string, InferCreator> inference_engines;
    std::unordered_map<std::string, PostprocCreator> postprocessors;
  };

  PluginRegistry() = default;

  Snapshot snapshot() const;
  void restore(Snapshot snapshot);

  template <class Map, class Creator>
  bool add(Map &map, std::string name, Creator creator);

  mutable std::shared_mutex m_mutex;
  std::unordered_map<std::string, PreprocCreator> m_preprocessors;
  std::unordered_map<std::string, InferCreator> m_inferenceEngines;
  std::unordered_map<std::string, PostprocCreator> m_postprocessors;

  friend class PluginManager;
};

} // namespace ai_core::dnn

#endif
