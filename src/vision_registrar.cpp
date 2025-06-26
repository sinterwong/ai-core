/**
 * @file vision_registrar.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "vision_registrar.hpp"

#include "fpr_cls.hpp"
#include "fpr_feat.hpp"
#include "nano_det.hpp"
#include "rtm_det.hpp"
#include "softmax_cls.hpp"
#include "type_safe_factory.hpp"
#include "vision.hpp"
#include "yolo_det.hpp"

#include "logger.hpp"

namespace ai_core::dnn::vision {
#define REGISTER_VISION_ALGO(AlgoName)                                         \
  VisionFactory::instance().registerCreator(                                   \
      #AlgoName,                                                               \
      [](const AlgoConstructParams &cparams) -> std::shared_ptr<VisionBase> {  \
        auto params = cparams.getParam<AlgoPostprocParams>("params");          \
        return std::make_shared<AlgoName>(params);                             \
      });                                                                      \
  LOG_INFOS << "Registered " #AlgoName " creator."

VisionRegistrar::VisionRegistrar() {
  REGISTER_VISION_ALGO(RTMDet);
  REGISTER_VISION_ALGO(Yolov11Det);
  REGISTER_VISION_ALGO(NanoDet);
  REGISTER_VISION_ALGO(SoftmaxCls);
  REGISTER_VISION_ALGO(FprCls);
  REGISTER_VISION_ALGO(FprFeature);
}
} // namespace ai_core::dnn::vision
