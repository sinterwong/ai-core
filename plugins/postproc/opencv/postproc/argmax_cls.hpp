#ifndef AI_CORE_INFERENCE_VISION_ARGMAX_CLS_HPP
#define AI_CORE_INFERENCE_VISION_ARGMAX_CLS_HPP

#include "frame_postproc_base.hpp"
namespace ai_core::dnn {
/**
 * @brief Takes the top-1 of an already-normalized score vector and passes the
 * score through untouched.
 *
 * Counterpart to @ref SoftmaxCls: use this when the model bakes the softmax
 * into its graph (ultralytics `*-cls` exports do — their output sums to 1).
 * Running SoftmaxCls on such an output softmaxes twice, which keeps the label
 * but collapses the confidence to near the 1/num_classes floor.
 */
class ArgmaxCls : public FramePostprocBase<GenericPostParams, false> {
public:
  explicit ArgmaxCls() {}

  virtual bool processTyped(const TensorData &, const FrameTransformContext &,
                            const GenericPostParams &,
                            AlgoOutput &) const override;

  virtual bool batchProcessTyped(const TensorData &,
                                 const std::vector<FrameTransformContext> &,
                                 const GenericPostParams &,
                                 std::vector<AlgoOutput> &) const override;

private:
  ClsRet processSingleItem(const float *scores, int num_classes,
                           bool keep_probs) const;
};
} // namespace ai_core::dnn

#endif
