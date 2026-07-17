/**
 * @file softmax_cls.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFERENCE_VISION_SOFTMAX_CLS_HPP
#define AI_CORE_INFERENCE_VISION_SOFTMAX_CLS_HPP

#include "frame_postproc_base.hpp"
namespace ai_core::dnn {
class SoftmaxCls : public FramePostprocBase<GenericPostParams, false> {
public:
  explicit SoftmaxCls() {}

  virtual bool processTyped(const TensorData &, const FrameTransformContext &,
                            const GenericPostParams &,
                            AlgoOutput &) const override;

  virtual bool batchProcessTyped(const TensorData &,
                                 const std::vector<FrameTransformContext> &,
                                 const GenericPostParams &,
                                 std::vector<AlgoOutput> &) const override;

private:
  ClsRet processSingleItem(const float *logits, int num_classes) const;
};
} // namespace ai_core::dnn

#endif
