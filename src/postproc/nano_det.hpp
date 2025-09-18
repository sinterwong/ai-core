/**
 * @file yoloDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_NANODET_DETECTION_HPP_
#define __INFERENCE_VISION_NANODET_DETECTION_HPP_

#include "anchor_det_post_base.hpp"
namespace ai_core::dnn {
class NanoDet : public IAnchorDetPostprocessor {
public:
  explicit NanoDet() {}

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const AnchorDetParams &, AlgoOutput &) const override;
};
} // namespace ai_core::dnn

#endif
