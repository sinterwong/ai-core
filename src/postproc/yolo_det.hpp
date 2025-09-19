/**
 * @file yoloDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_YOLOV11_DETECTION_HPP_
#define __INFERENCE_VISION_YOLOV11_DETECTION_HPP_

#include "ai_core/preproc_types.hpp"
#include "anchor_det_post_base.hpp"
namespace ai_core::dnn {
class Yolov11Det : public IAnchorDetPostprocessor {
public:
  explicit Yolov11Det() {}

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const AnchorDetParams &, AlgoOutput &) const override;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const AnchorDetParams &,
                            std::vector<AlgoOutput> &) const override;

private:
  std::vector<BBox> processRawOutput(const cv::Mat &transposedOutput,
                                     const Shape &inputShape,
                                     const FrameTransformContext &prepArgs,
                                     const AnchorDetParams &postArgs,
                                     int numClasses) const;
};
} // namespace ai_core::dnn

#endif
