/**
 * @file semantic_seg.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_VISION_SEMANTIC_SEG_HPP
#define AI_CORE_VISION_SEMANTIC_SEG_HPP
#include "confidence_filter_post_base.hpp"

namespace ai_core::dnn {
class SemanticSeg : public IConfidencePostprocessor {
public:
  explicit SemanticSeg() {}

  virtual bool process(const TensorData &model_output,
                       const FrameTransformContext &prep_args,
                       const ConfidenceFilterParams &params,
                       AlgoOutput &algo_output) const override;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const ConfidenceFilterParams &,
                            std::vector<AlgoOutput> &) const override;

private:
  SegRet processSingleItem(const float *data, int num_classes, int height,
                           int width, const FrameTransformContext &prep_args,
                           const ConfidenceFilterParams &post_args) const;
};
} // namespace ai_core::dnn

#endif
