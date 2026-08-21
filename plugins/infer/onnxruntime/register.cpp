#include "ai_core/plugin_manager.hpp"
#include "ai_core/version.hpp"
#include "ort/ort_infer.hpp"

#include <memory>

using namespace ai_core;
using namespace ai_core::dnn;

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(PluginRegistry &registry, PluginInfo &info) {
  info = {.name = "ai_core.infer.onnxruntime",
          .version = AI_CORE_SEMVER_STR,
          .provider = "ai-core",
          .description = "ONNX Runtime inference backend",
          .capabilities = {"infer", "backend:onnxruntime"}};
  return registry.registerInferenceEngine(
      "OrtAlgoInference", [](const DataPacket &params) {
        return std::make_shared<OrtAlgoInference>(params);
      });
}
