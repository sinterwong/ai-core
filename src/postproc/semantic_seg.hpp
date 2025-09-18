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

  virtual bool process(const TensorData &modelOutput,
                       const FrameTransformContext &prepArgs,
                       const ConfidenceFilterParams &params,
                       AlgoOutput &algoOutput) const override;
};
} // namespace ai_core::dnn

#endif
