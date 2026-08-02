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
#include "vision_util.hpp"

#include "ai_core/opencv_interop.hpp"
#include "ai_core/output_types.hpp"
#include "ai_core/preprocess_types.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ai_core::utils {

PixelFormatPlan planPixelFormat(ImagePixelFormat from, ImagePixelFormat to) {
  using F = ImagePixelFormat;
  PixelFormatPlan plan;

  if (from == to) {
    return plan;
  }

  // Same channel count => pure channel permutation, foldable into the caller's
  // normalization pass. Alpha (index 3) stays put.
  if (channelCount(from) == channelCount(to)) {
    plan.channel_map = {2, 1, 0, 3};
    return plan;
  }

  plan.needs_cvt_color = true;
  switch (from) {
  case F::GRAY8:
    switch (to) {
    case F::BGR888:
      plan.cvt_code = cv::COLOR_GRAY2BGR;
      return plan;
    case F::RGB888:
      plan.cvt_code = cv::COLOR_GRAY2RGB;
      return plan;
    // Gray has no channel order, so BGRA and RGBA are the same expansion.
    case F::BGRA8888:
    case F::RGBA8888:
      plan.cvt_code = cv::COLOR_GRAY2BGRA;
      return plan;
    default:
      break;
    }
    break;
  case F::BGR888:
    switch (to) {
    case F::GRAY8:
      plan.cvt_code = cv::COLOR_BGR2GRAY;
      return plan;
    case F::BGRA8888:
      plan.cvt_code = cv::COLOR_BGR2BGRA;
      return plan;
    case F::RGBA8888:
      plan.cvt_code = cv::COLOR_BGR2RGBA;
      return plan;
    default:
      break;
    }
    break;
  case F::RGB888:
    switch (to) {
    case F::GRAY8:
      plan.cvt_code = cv::COLOR_RGB2GRAY;
      return plan;
    case F::RGBA8888:
      plan.cvt_code = cv::COLOR_RGB2RGBA;
      return plan;
    case F::BGRA8888:
      plan.cvt_code = cv::COLOR_RGB2BGRA;
      return plan;
    default:
      break;
    }
    break;
  case F::BGRA8888:
    switch (to) {
    case F::GRAY8:
      plan.cvt_code = cv::COLOR_BGRA2GRAY;
      return plan;
    case F::BGR888:
      plan.cvt_code = cv::COLOR_BGRA2BGR;
      return plan;
    case F::RGB888:
      plan.cvt_code = cv::COLOR_BGRA2RGB;
      return plan;
    default:
      break;
    }
    break;
  case F::RGBA8888:
    switch (to) {
    case F::GRAY8:
      plan.cvt_code = cv::COLOR_RGBA2GRAY;
      return plan;
    case F::RGB888:
      plan.cvt_code = cv::COLOR_RGBA2RGB;
      return plan;
    case F::BGR888:
      plan.cvt_code = cv::COLOR_RGBA2BGR;
      return plan;
    default:
      break;
    }
    break;
  }

  throw std::invalid_argument(
      "Unsupported pixel format conversion: " +
      std::to_string(static_cast<int>(from)) + " -> " +
      std::to_string(static_cast<int>(to)));
}

cv::Mat convertPixelFormat(const cv::Mat &src, ImagePixelFormat from,
                           ImagePixelFormat to) {
  const PixelFormatPlan plan = planPixelFormat(from, to);
  if (plan.isIdentity()) {
    return src;
  }
  cv::Mat dst;
  if (plan.needs_cvt_color) {
    cv::cvtColor(src, dst, plan.cvt_code);
  } else {
    // Same channel count: a channel reversal. BGR2RGB and RGB2BGR are the same
    // operation, likewise BGRA2RGBA / RGBA2BGRA.
    cv::cvtColor(src, dst,
                 src.channels() == 4 ? cv::COLOR_BGRA2RGBA
                                     : cv::COLOR_BGR2RGB);
  }
  return dst;
}

float calculateIoU(const BBox &bbox1, const BBox &bbox2) {
  const auto &b1_rect = bbox1.rect;
  const auto &b2_rect = bbox2.rect;

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
      boxes.push_back(interop::toCv(result.rect));
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