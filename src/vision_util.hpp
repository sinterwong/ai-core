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

#include "ai_core/types/algo_output_types.hpp"
#include "ai_core/types/infer_common_types.hpp"

namespace ai_core::utils {

std::pair<float, float> scaleRatio(Shape const &originShape,
                                   Shape const &inputShape, bool isScale);

float calculateIoU(const BBox &bbox1, const BBox &bbox2);

std::vector<BBox> NMS(const std::vector<BBox> &results, float nmsThre,
                      float confThre);

Shape escaleResizeWithPad(const cv::Mat &src, cv::Mat &dst, int targetWidth,
                          int targetHeight, const cv::Scalar &pad);
} // namespace ai_core::utils
#endif