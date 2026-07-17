/**
 * @file fpr_cls.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFERENCE_VISION_FPR_CLS_HPP
#define AI_CORE_INFERENCE_VISION_FPR_CLS_HPP

#include "frame_postproc_base.hpp"
namespace ai_core::dnn {
class FprCls : public FramePostprocBase<GenericPostParams, false> {
public:
  explicit FprCls() {}

  virtual bool processTyped(const TensorData &, const FrameTransformContext &,
                       const GenericPostParams &, AlgoOutput &) const override;

  virtual bool batchProcessTyped(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const GenericPostParams &,
                            std::vector<AlgoOutput> &) const override;

private:
  FprClsRet processSingleItem(const float *scores_data, int num_classes,
                              const float *birads_data, int num_birads) const;
};
} // namespace ai_core::dnn

#endif
