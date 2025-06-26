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

#include "ai_core/types/algo_data_types.hpp" // For AlgoOutput, AlgoPostprocParams, FramePreprocessArg (via algo_input_types)
#include "ai_core/types/model_output.hpp"  // For ModelOutput
#include "vision.hpp"                      // Internal header
namespace ai_core::dnn::vision {
class FprCls : public VisionBase {
public:
  explicit FprCls(const AlgoPostprocParams &params) : mParams(params) {}

  virtual bool processOutput(const ModelOutput &, const FramePreprocessArg &,
                             AlgoOutput &) override;

private:
  AlgoPostprocParams mParams;
};
} // namespace ai_core::dnn::vision

#endif
