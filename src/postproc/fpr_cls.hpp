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

  virtual bool process(const TensorData &, const FramePreprocessArg &,
                       AlgoOutput &, const GenericPostParams &) const override;
};
} // namespace ai_core::dnn

#endif
