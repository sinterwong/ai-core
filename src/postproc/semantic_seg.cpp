/**
 * @file fpr_cls.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "semantic_seg.hpp"
#include "ai_core/postproc_types.hpp"
#include "logger.hpp"
#include "vision_util.hpp"
#include <opencv2/opencv.hpp>

namespace ai_core::dnn {
bool SemanticSeg::process(const TensorData &modelOutput,
                          const FramePreprocessArg &prepArgs,
                          AlgoOutput &algoOutput,
                          const ConfidenceFilterParams &postArgs) const {
  const auto &featMapOutputName = postArgs.outputNames.at(0);
  const auto &featMapOutput = modelOutput.datas.at(featMapOutputName);
  const auto &featMapShape = modelOutput.shapes.at(featMapOutputName);

  // NCHW
  const int numClasses = featMapShape.at(featMapShape.size() - 3);

  if (numClasses > 256) {
    LOG_ERRORS << "Too many classes for CV_8UC1 mask.";
    return false;
  }

  const int height = featMapShape.at(featMapShape.size() - 2);
  const int width = featMapShape.at(featMapShape.size() - 1);
  const size_t channelStep = static_cast<size_t>(height) * width;
  const float *data = featMapOutput.getHostPtr<float>();

  cv::Mat classMap(height, width, CV_8UC1);

  if (numClasses == 1) {
    // 将原始数据包装成cv::Mat
    cv::Mat probMap(height, width, CV_32FC1, const_cast<float *>(data));
    // 大于阈值的设为1，否则为0
    cv::threshold(probMap, classMap, postArgs.condThre, 1, cv::THRESH_BINARY);
  } else {
    cv::Mat maxProbs(height, width, CV_32F, const_cast<float *>(data));
    classMap.setTo(0); // 默认类别为0 (背景)

    // 从第二个通道开始遍历，更新 maxProbs 和 classMap
    for (int c = 1; c < numClasses; ++c) {
      cv::Mat currentProbs(height, width, CV_32F,
                           const_cast<float *>(data + c * channelStep));
      cv::Mat updateMask;
      // 比较当前通道的概率是否大于已记录的最大概率
      cv::compare(currentProbs, maxProbs, updateMask, cv::CMP_GT);

      currentProbs.copyTo(maxProbs, updateMask);
      classMap.setTo(c, updateMask);
    }

    // 将最大概率小于阈值的像素类别置为0
    cv::Mat lowConfidenceMask;
    cv::compare(maxProbs, postArgs.condThre, lowConfidenceMask, cv::CMP_LT);
    classMap.setTo(0, lowConfidenceMask);
  }

  SegRet segRet;
  segRet.clsToContours.clear();

  auto [scaleX, scaleY] = utils::scaleRatio(
      prepArgs.originShape, prepArgs.modelInputShape, prepArgs.isEqualScale);

  if (scaleX <= 0.0f || scaleY <= 0.0f) {
    LOG_ERRORS << "Invalid scale factors detected: scaleX=" << scaleX
               << ", scaleY=" << scaleY;
    return false;
  }

  int startClass = 1;
  int endClass = (numClasses == 1) ? 2 : numClasses;

  for (int c = startClass; c < endClass; ++c) {
    cv::Mat classMask;
    cv::compare(classMap, c, classMask, cv::CMP_EQ);

    if (cv::countNonZero(classMask) == 0) {
      continue;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(classMask, contours, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
      continue;
    }

    const float offsetX = prepArgs.roi->x - prepArgs.leftPad / scaleX;
    const float offsetY = prepArgs.roi->y - prepArgs.topPad / scaleY;

    for (const auto &contour : contours) {
      std::vector<cv::Point> transformedContour;
      transformedContour.reserve(contour.size());
      std::transform(contour.begin(), contour.end(),
                     std::back_inserter(transformedContour),
                     [&](const cv::Point &pt) -> cv::Point {
                       float originalX =
                           static_cast<float>(pt.x) / scaleX + offsetX;
                       float originalY =
                           static_cast<float>(pt.y) / scaleY + offsetY;
                       return cv::Point(cvRound(originalX), cvRound(originalY));
                     });

      segRet.clsToContours[c].emplace_back(std::move(transformedContour));
    }
  }

  algoOutput.setParams(segRet);
  return true;
}
} // namespace ai_core::dnn
