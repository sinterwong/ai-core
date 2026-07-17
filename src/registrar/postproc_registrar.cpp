/**
 * @file postproc_registrar.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "postproc_registrar.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/plugin_registrar.hpp"
#include "postproc/fpr_cls.hpp"
#include "postproc/nano_det.hpp"
#include "postproc/ocr_reco.hpp"
#include "postproc/raw_output.hpp"
#include "postproc/rtm_det.hpp"
#include "postproc/semantic_seg.hpp"
#include "postproc/softmax_cls.hpp"
#include "postproc/unet_dual_out_seg.hpp"
#include "postproc/yolo_det.hpp"

namespace ai_core::dnn {

DefaultPostprocPluginRegistrar::DefaultPostprocPluginRegistrar() {
  REGISTER_POSTPROCESS_ALGO(Yolov11Det);
  REGISTER_POSTPROCESS_ALGO(RTMDet);
  REGISTER_POSTPROCESS_ALGO(NanoDet);
  REGISTER_POSTPROCESS_ALGO(SoftmaxCls);
  REGISTER_POSTPROCESS_ALGO(FprCls);
  REGISTER_POSTPROCESS_ALGO(RawModelOutput);
  REGISTER_POSTPROCESS_ALGO(OCRReco);
  REGISTER_POSTPROCESS_ALGO(UNetDualOutputSeg);
  REGISTER_POSTPROCESS_ALGO(SemanticSeg);
  LOG_INFO_S << "Default postprocess plugins registered." << std::endl;
}
} // namespace ai_core::dnn
