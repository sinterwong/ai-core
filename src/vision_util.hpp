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
#ifndef AI_CORE_INFERENCE_VISION_UTILS_HPP
#define AI_CORE_INFERENCE_VISION_UTILS_HPP

#include "ai_core/output_types.hpp"
#include "ai_core/common_types.hpp"
#include <opencv2/core/types.hpp>

namespace ai_core::utils {

std::pair<float, float> scaleRatio(Shape const &origin_shape,
                                   Shape const &input_shape, bool is_scale);

float calculateIoU(const BBox &bbox1, const BBox &bbox2);

std::vector<BBox> nms(const std::vector<BBox> &results, float nms_thre,
                      float conf_thre);

Shape escaleResizeWithPad(const cv::Mat &src, cv::Mat &dst, int target_width,
                          int target_height, const cv::Scalar &pad);

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