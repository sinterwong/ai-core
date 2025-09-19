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
#ifndef __INFERENCE_VISION_FPR_CLS_HPP_
#define __INFERENCE_VISION_FPR_CLS_HPP_

#include "cv_generic_post_base.hpp"
namespace ai_core::dnn {
class FprCls : public ICVGenericPostprocessor {
public:
  explicit FprCls() {}

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const GenericPostParams &, AlgoOutput &) const override;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const GenericPostParams &,
                            std::vector<AlgoOutput> &) const override;

private:
  FprClsRet processSingleItem(const float *scoresData, int numClasses,
                              const float *biradsData, int numBirads) const;
};
} // namespace ai_core::dnn

#endif
