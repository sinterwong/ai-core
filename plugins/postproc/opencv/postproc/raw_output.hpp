/**
 * @file raw_output.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFERENCE_VISION_RAW_MODEL_OUTPUT_HPP
#define AI_CORE_INFERENCE_VISION_RAW_MODEL_OUTPUT_HPP

#include "frame_postproc_base.hpp"
namespace ai_core::dnn {
class RawModelOutput : public FramePostprocBase<GenericPostParams, false> {
public:
  explicit RawModelOutput() {}

  virtual bool processTyped(const TensorData &, const FrameTransformContext &,
                            const GenericPostParams &,
                            AlgoOutput &) const override;

  virtual bool batchProcessTyped(const TensorData &,
                                 const std::vector<FrameTransformContext> &,
                                 const GenericPostParams &,
                                 std::vector<AlgoOutput> &) const override;
};
} // namespace ai_core::dnn

#endif
