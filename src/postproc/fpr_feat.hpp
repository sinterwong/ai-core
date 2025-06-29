/**
 * @file fpr_feat.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_FPR_FEATURE_HPP_
#define __INFERENCE_VISION_FPR_FEATURE_HPP_

#include "postproc_base.hpp"
namespace ai_core::dnn {
class FprFeature : public PostprocssBase {
public:
  explicit FprFeature() {}

  virtual bool process(const TensorData &, AlgoPreprocParams &, AlgoOutput &,
                       AlgoPostprocParams &) override;
};
} // namespace ai_core::dnn

#endif
