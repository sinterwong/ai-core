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
#ifndef AI_CORE_INFERENCE_VISION_RTM_DETECTION_HPP
#define AI_CORE_INFERENCE_VISION_RTM_DETECTION_HPP

#include "frame_postproc_base.hpp"
namespace ai_core::dnn {
class RTMDet : public FramePostprocBase<AnchorDetParams, true> {
public:
  explicit RTMDet() {}

  virtual bool processTyped(const TensorData &, const FrameTransformContext &,
                            const AnchorDetParams &,
                            AlgoOutput &) const override;

  virtual bool batchProcessTyped(const TensorData &,
                                 const std::vector<FrameTransformContext> &,
                                 const AnchorDetParams &,
                                 std::vector<AlgoOutput> &) const override;

private:
  DetRet processSingle(const float *det_data_ptr, const float *cls_data_ptr,
                       int anchor_num, int num_classes,
                       const FrameTransformContext &prep_args,
                       const AnchorDetParams &post_args) const;
};
} // namespace ai_core::dnn

#endif
