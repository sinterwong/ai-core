/**
 * @file vision_util.cpp
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
#include "ai_core/preprocess_types.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>

namespace ai_core::utils {

std::pair<float, float> scaleRatio(Shape const &origin_shape,
                                   Shape const &input_shape, bool is_scale) {
  float rw, rh;
  if (is_scale) {
    rw = std::min(static_cast<float>(input_shape.w) / origin_shape.w,
                  static_cast<float>(input_shape.h) / origin_shape.h);
    rh = rw;
  } else {
    rw = static_cast<float>(input_shape.w) / origin_shape.w;
    rh = static_cast<float>(input_shape.h) / origin_shape.h;
  }
  return std::make_pair(rw, rh);
}

float calculateIoU(const BBox &bbox1, const BBox &bbox2) {
  const auto &b1_rect = *bbox1.rect;
  const auto &b2_rect = *bbox2.rect;

  float x1 = std::max(b1_rect.x, b2_rect.x);
  float y1 = std::max(b1_rect.y, b2_rect.y);
  float x2 = std::min(b1_rect.x + b1_rect.width, b2_rect.x + b2_rect.width);
  float y2 = std::min(b1_rect.y + b1_rect.height, b2_rect.y + b2_rect.height);

  if (x2 < x1 || y2 < y1) {
    return 0.0f;
  }

  float intersection = (x2 - x1) * (y2 - y1);
  float area1 = b1_rect.area();
  float area2 = b2_rect.area();
  float union_area = area1 + area2 - intersection;

  return intersection / union_area;
}

std::vector<BBox> nms(const std::vector<BBox> &results, float nms_thre,
                      float conf_thre) {
  std::unordered_map<int, std::vector<BBox>> class_results;
  for (const auto &result : results) {
    class_results[result.label].push_back(result);
  }

  std::vector<BBox> nms_results;
  for (auto &pair : class_results) {
    auto &class_result = pair.second;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;

    for (const auto &result : class_result) {
      boxes.push_back(*result.rect);
      scores.push_back(result.score);
    }

    cv::dnn::NMSBoxes(boxes, scores, conf_thre, nms_thre, indices);

    for (int idx : indices) {
      nms_results.push_back(class_result[idx]);
    }
  }

  return nms_results;
}

Shape escaleResizeWithPad(const cv::Mat &src, cv::Mat &dst, int target_width,
                          int target_height, const cv::Scalar &pad) {
  float scale = std::min(static_cast<float>(target_width) / src.cols,
                         static_cast<float>(target_height) / src.rows);
  cv::Size new_size(static_cast<int>(src.cols * scale),
                    static_cast<int>(src.rows * scale));
  cv::resize(src, dst, new_size, 0, 0, cv::INTER_LINEAR);
  Shape pad_ret;
  pad_ret.h = (target_height - dst.rows) / 2;
  pad_ret.w = (target_width - dst.cols) / 2;
  int bottom_pad = target_height - dst.rows - pad_ret.h;
  int right_pad = target_width - dst.cols - pad_ret.w;
  cv::copyMakeBorder(dst, dst, pad_ret.h, bottom_pad, pad_ret.w, right_pad,
                     cv::BORDER_CONSTANT, pad);

  return pad_ret;
}

} // namespace ai_core::utils
#endif