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

#include "postproc_base.hpp"
namespace ai_core::dnn {
class Yolov11Det : public PostprocssBase {
public:
  explicit Yolov11Det() {}

  virtual bool process(const TensorData &, AlgoPreprocParams &, AlgoOutput &,
                       AlgoPostprocParams &) override;

private:
  std::vector<BBox> processRawOutput(const cv::Mat &transposedOutput,
                                     const Shape &inputShape,
                                     const FramePreprocessArg &prepArgs,
                                     const AnchorDetParams &postArgs,
                                     int numClasses);
};
} // namespace ai_core::dnn

#endif
