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

#include "postproc/fpr_cls.hpp"
#include "postproc/fpr_feat.hpp"
#include "postproc/nano_det.hpp"
#include "postproc/rtm_det.hpp"
#include "postproc/softmax_cls.hpp"
#include "postproc/yolo_det.hpp"
#include "postproc_base.hpp"
#include "type_safe_factory.hpp"

#include "logger.hpp"

namespace ai_core::dnn {
#define REGISTER_POSTPROCESS_ALGO(AlgoName)                                    \
  PostprocFactory::instance().registerCreator(                                 \
      #AlgoName,                                                               \
      [](const AlgoConstructParams &cparams)                                   \
          -> std::shared_ptr<PostprocssBase> {                                 \
        return std::make_shared<AlgoName>();                                   \
      });                                                                      \
  LOG_INFOS << "Registered " #AlgoName " creator."

PostprocessRegistrar::PostprocessRegistrar() {
  REGISTER_POSTPROCESS_ALGO(RTMDet);
  REGISTER_POSTPROCESS_ALGO(Yolov11Det);
  REGISTER_POSTPROCESS_ALGO(NanoDet);
  REGISTER_POSTPROCESS_ALGO(SoftmaxCls);
  REGISTER_POSTPROCESS_ALGO(FprCls);
  REGISTER_POSTPROCESS_ALGO(FprFeature);
}
} // namespace ai_core::dnn
