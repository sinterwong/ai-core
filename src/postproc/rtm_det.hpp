/**
 * @file rtmDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_RTM_DETECTION_HPP_
#define __INFERENCE_VISION_RTM_DETECTION_HPP_

#include "postproc_base.hpp"
namespace ai_core::dnn {
class RTMDet : public PostprocssBase {
public:
  explicit RTMDet() {}

  virtual bool process(const TensorData &, AlgoPreprocParams &, AlgoOutput &,
                       AlgoPostprocParams &) override;
};
} // namespace ai_core::dnn

#endif
