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
#ifndef __INFERENCE_VISION_SOFTMAX_CLS_HPP_
#define __INFERENCE_VISION_SOFTMAX_CLS_HPP_

#include "cv_generic_post_base.hpp"
namespace ai_core::dnn {
class SoftmaxCls : public ICVGenericPostprocessor {
public:
  explicit SoftmaxCls() {}

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const GenericPostParams &, AlgoOutput &) const override;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const GenericPostParams &,
                            std::vector<AlgoOutput> &) const override;

private:
  ClsRet processSingleItem(const float *logits, int numClasses) const;
};
} // namespace ai_core::dnn

#endif
