#include "ai_core/plugin_manager.hpp"
#include "ai_core/version.hpp"
#include "preproc/cpu_generic_preprocess.hpp"
#include "preproc/frame_with_mask_prep.hpp"

#include <memory>

using namespace ai_core;
using namespace ai_core::dnn;

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(PluginRegistry &registry, PluginInfo &info) {
  info = {.name = "ai_core.preproc.opencv",
          .version = AI_CORE_SEMVER_STR,
          .provider = "ai-core",
          .description = "OpenCV CPU preprocessing",
          .capabilities = {"preproc", "backend:cpu", "framework:opencv"}};
  bool ok = true;
  ok &= registry.registerPreprocessor(
      "CpuGenericPreprocess", [](const DataPacket &) {
        return std::make_shared<CpuGenericPreprocess>();
      });
  ok &= registry.registerPreprocessor(
      "FrameWithMaskPreprocess", [](const DataPacket &) {
        return std::make_shared<FrameWithMaskPreprocess>();
      });
  return ok;
}
