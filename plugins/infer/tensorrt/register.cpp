#include "ai_core/plugin_manager.hpp"
#include "ai_core/version.hpp"
#include "preproc/cuda_generic_preprocess.hpp"
#include "trt/trt_infer.hpp"

#include <memory>

using namespace ai_core;
using namespace ai_core::dnn;

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(PluginRegistry &registry, PluginInfo &info) {
  info = {.name = "ai_core.infer.tensorrt",
          .version = AI_CORE_SEMVER_STR,
          .provider = "ai-core",
          .description = "TensorRT CUDA inference and preprocessing",
          .capabilities = {"infer", "preproc", "backend:cuda",
                           "framework:tensorrt"}};
  bool ok = registry.registerInferenceEngine(
      "TrtAlgoInference", [](const DataPacket &params) {
        return std::make_shared<TrtAlgoInference>(params);
      });
  ok &= registry.registerPreprocessor(
      "CudaGenericPreprocess", [](const DataPacket &) {
        return std::make_shared<CudaGenericPreprocess>();
      });
  return ok;
}
