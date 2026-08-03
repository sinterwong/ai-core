#include "ai_core/plugin_manager.hpp"
#include "postproc/argmax_cls.hpp"
#include "postproc/fpr_cls.hpp"
#include "postproc/nano_det.hpp"
#include "postproc/ocr_reco.hpp"
#include "postproc/raw_output.hpp"
#include "postproc/rtm_det.hpp"
#include "postproc/semantic_seg.hpp"
#include "postproc/softmax_cls.hpp"
#include "postproc/unet_dual_out_seg.hpp"
#include "postproc/yolo_det.hpp"

#include <memory>

using namespace ai_core;
using namespace ai_core::dnn;

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(PluginRegistry &registry, PluginInfo &info) {
  info = {.name = "ai_core.postproc.opencv",
          .version = "2.1.0",
          .provider = "ai-core",
          .description = "OpenCV model postprocessing",
          .capabilities = {"postproc", "backend:cpu", "framework:opencv"}};
  bool ok = true;
#define AI_CORE_REGISTER_POSTPROCESSOR(Type)                                  \
  ok &= registry.registerPostprocessor(#Type, [](const DataPacket &) {         \
    return std::make_shared<::ai_core::dnn::Type>();                           \
  })
  AI_CORE_REGISTER_POSTPROCESSOR(Yolov11Det);
  AI_CORE_REGISTER_POSTPROCESSOR(RTMDet);
  AI_CORE_REGISTER_POSTPROCESSOR(NanoDet);
  AI_CORE_REGISTER_POSTPROCESSOR(SoftmaxCls);
  AI_CORE_REGISTER_POSTPROCESSOR(ArgmaxCls);
  AI_CORE_REGISTER_POSTPROCESSOR(FprCls);
  AI_CORE_REGISTER_POSTPROCESSOR(RawModelOutput);
  AI_CORE_REGISTER_POSTPROCESSOR(OCRReco);
  AI_CORE_REGISTER_POSTPROCESSOR(UNetDualOutputSeg);
  AI_CORE_REGISTER_POSTPROCESSOR(SemanticSeg);
#undef AI_CORE_REGISTER_POSTPROCESSOR
  return ok;
}
