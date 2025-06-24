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

#include "infer_types.hpp"
#include "vision.hpp"
namespace infer::dnn::vision {
class FprFeature : public VisionBase {
public:
  explicit FprFeature(const AlgoPostprocParams &params) : mParams(params) {}

  virtual bool processOutput(const ModelOutput &, const FramePreprocessArg &,
                             AlgoOutput &) override;

private:
  AlgoPostprocParams mParams;
};
} // namespace infer::dnn::vision

#endif
