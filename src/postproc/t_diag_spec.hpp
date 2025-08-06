/**
 * @file t_diag_spec.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_T_DIAG_SPEC_HPP_
#define __INFERENCE_VISION_T_DIAG_SPEC_HPP_

#include "cv_generic_post_base.hpp"
namespace ai_core::dnn {
class TDiagSpecPostproc : public ICVGenericPostprocessor {
public:
  explicit TDiagSpecPostproc() {}

  virtual bool process(const TensorData &, const FramePreprocessArg &,
                       AlgoOutput &, const GenericPostParams &) const override;
};
} // namespace ai_core::dnn

#endif
