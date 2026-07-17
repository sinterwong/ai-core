/**
 * @file unet_dual_out_seg.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFERENCE_VISION_UNET_DAUL_OUTPUTS_HPP
#define AI_CORE_INFERENCE_VISION_UNET_DAUL_OUTPUTS_HPP

#include "cv_generic_post_base.hpp"
namespace ai_core::dnn {
class UNetDualOutputSeg : public ICVGenericPostprocessor {
public:
  explicit UNetDualOutputSeg() {}

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const GenericPostParams &, AlgoOutput &) const override;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const GenericPostParams &,
                            std::vector<AlgoOutput> &) const override;

private:
  DualRawSegRet processSingleItem(const float *prob_data,
                                  const std::vector<int> &prob_shape,
                                  const float *mask_data,
                                  const std::vector<int> &mask_shape,
                                  const FrameTransformContext &prep_args) const;
};
} // namespace ai_core::dnn

#endif
