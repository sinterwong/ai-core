/**
 * @file unet_daul_out_seg.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_UNET_DAUL_OUTPUTS_HPP_
#define __INFERENCE_VISION_UNET_DAUL_OUTPUTS_HPP_

#include "cv_generic_post_base.hpp"
namespace ai_core::dnn {
class UNetDaulOutputSeg : public ICVGenericPostprocessor {
public:
  explicit UNetDaulOutputSeg() {}

  virtual bool process(const TensorData &, const FramePreprocessArg &,
                       AlgoOutput &, const GenericPostParams &) const override;
};
} // namespace ai_core::dnn

#endif
