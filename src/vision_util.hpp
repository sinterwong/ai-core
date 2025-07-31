/**
 * @file vision_util.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_UTILS_HPP_
#define __INFERENCE_VISION_UTILS_HPP_

#include "ai_core/algo_output_types.hpp"
#include "ai_core/infer_common_types.hpp"
#include <opencv2/core/types.hpp>

namespace ai_core::utils {

std::pair<float, float> scaleRatio(Shape const &originShape,
                                   Shape const &inputShape, bool isScale);

float calculateIoU(const BBox &bbox1, const BBox &bbox2);

std::vector<BBox> NMS(const std::vector<BBox> &results, float nmsThre,
                      float confThre);

Shape escaleResizeWithPad(const cv::Mat &src, cv::Mat &dst, int targetWidth,
                          int targetHeight, const cv::Scalar &pad);

template <typename T>
cv::Scalar createScalarFromVector(const std::vector<T> &values) {
  cv::Scalar s;
  size_t n = std::min(values.size(), (size_t)4);
  for (size_t i = 0; i < n; ++i) {
    s[i] = static_cast<double>(values[i]);
  }
  return s;
}
} // namespace ai_core::utils
#endif