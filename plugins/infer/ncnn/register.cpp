#include "ai_core/plugin_manager.hpp"
#include "ai_core/version.hpp"
#include "ncnn/ncnn_infer.hpp"

#include <memory>

using namespace ai_core;
using namespace ai_core::dnn;

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(PluginRegistry &registry, PluginInfo &info) {
  info = {.name = "ai_core.infer.ncnn",
          .version = AI_CORE_SEMVER_STR,
          .provider = "ai-core",
          .description = "NCNN inference backend",
          .capabilities = {"infer", "backend:ncnn"}};
  return registry.registerInferenceEngine(
      "NCNNAlgoInference", [](const DataPacket &params) {
        return std::make_shared<NCNNAlgoInference>(params);
      });
}
