#include "ai_core/plugin_manager.hpp"
#include "ncnn/ncnn_infer.hpp"

#include <memory>

using namespace ai_core;
using namespace ai_core::dnn;

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(PluginRegistry &registry, PluginInfo &info) {
  info = {"ai_core.infer.ncnn", "2.1.0", "ai-core"};
  return registry.registerInferenceEngine(
      "NCNNAlgoInference", [](const DataPacket &params) {
        return std::make_shared<NCNNAlgoInference>(params);
      });
}
