#ifndef AI_CORE_VISION_SEMANTIC_SEG_HPP
#define AI_CORE_VISION_SEMANTIC_SEG_HPP
#include "frame_postproc_base.hpp"

namespace ai_core::dnn {
class SemanticSeg : public FramePostprocBase<ConfidenceFilterParams, true> {
public:
  explicit SemanticSeg() {}

  virtual bool processTyped(const TensorData &model_output,
                            const FrameTransformContext &prep_args,
                            const ConfidenceFilterParams &params,
                            AlgoOutput &algo_output) const override;

  virtual bool batchProcessTyped(const TensorData &,
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
