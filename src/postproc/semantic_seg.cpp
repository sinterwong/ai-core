/**
 * @file semantic_seg.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-19
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
                          const FrameTransformContext &prepArgs,
                          const ConfidenceFilterParams &postArgs,
                          AlgoOutput &algoOutput) const {
  const auto &featMapOutputName = postArgs.outputNames.at(0);
  const auto &featMapOutput = modelOutput.datas.at(featMapOutputName);
  const auto &featMapShape = modelOutput.shapes.at(featMapOutputName);

  const int numClasses = featMapShape.at(featMapShape.size() - 3);
  const int height = featMapShape.at(featMapShape.size() - 2);
  const int width = featMapShape.at(featMapShape.size() - 1);

  if (numClasses > 256) {
    LOG_ERRORS << "Too many classes for CV_8UC1 mask.";
    return false;
  }

  const float *data = featMapOutput.getHostPtr<float>();
  SegRet segRet =
      processSingleItem(data, numClasses, height, width, prepArgs, postArgs);

  algoOutput.setParams(segRet);
  return true;
}

bool SemanticSeg::batchProcess(
    const TensorData &modelOutput,
    const std::vector<FrameTransformContext> &prepArgs,
    const ConfidenceFilterParams &postArgs,
    std::vector<AlgoOutput> &algoOutput) const {
  const auto &featMapOutputName = postArgs.outputNames.at(0);
  const auto &featMapOutput = modelOutput.datas.at(featMapOutputName);
  const auto &featMapShape = modelOutput.shapes.at(featMapOutputName);

  if (featMapShape.size() != 4) {
    LOG_ERRORS << "Expected a 4D tensor for batch processing (NCHW), but got "
               << featMapShape.size() << " dimensions.";
    return false;
  }

  const int batchSize = featMapShape.at(0);
  const int numClasses = featMapShape.at(1);
  const int height = featMapShape.at(2);
  const int width = featMapShape.at(3);

  if (numClasses > 256) {
    LOG_ERRORS << "Too many classes for CV_8UC1 mask.";
    return false;
  }

  if (batchSize != prepArgs.size()) {
    LOG_ERRORS << "Batch size mismatch between model output (" << batchSize
               << ") and preprocessing arguments (" << prepArgs.size() << ").";
    return false;
  }

  const float *baseData = featMapOutput.getHostPtr<float>();
  const size_t itemStep = static_cast<size_t>(numClasses) * height * width;

  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    const float *currentItemData = baseData + i * itemStep;
    const FrameTransformContext &currentItemPrepArgs = prepArgs[i];

    SegRet segRet = processSingleItem(currentItemData, numClasses, height,
                                      width, currentItemPrepArgs, postArgs);

    algoOutput[i].setParams(segRet);
  }

  return true;
}

SegRet
SemanticSeg::processSingleItem(const float *data, int numClasses, int height,
                               int width, const FrameTransformContext &prepArgs,
                               const ConfidenceFilterParams &postArgs) const {
  const size_t channelStep = static_cast<size_t>(height) * width;
  cv::Mat classMap(height, width, CV_8UC1);

  if (numClasses == 1) {
    cv::Mat probMap(height, width, CV_32FC1, const_cast<float *>(data));
    // 大于阈值的设为1，否则为0
    cv::threshold(probMap, classMap, postArgs.condThre, 1, cv::THRESH_BINARY);
    classMap.convertTo(classMap, CV_8U); // 确保是 8 位
  } else {
    // 第一个通道作为初始最大概率图
    cv::Mat maxProbs(height, width, CV_32F, const_cast<float *>(data));
    classMap.setTo(0); // 默认类别为0 (背景)

    // 从第二个通道开始遍历，更新 maxProbs 和 classMap
    for (int c = 1; c < numClasses; ++c) {
      cv::Mat currentProbs(height, width, CV_32F,
                           const_cast<float *>(data + c * channelStep));
      cv::Mat updateMask;
      cv::compare(currentProbs, maxProbs, updateMask, cv::CMP_GT);

      currentProbs.copyTo(maxProbs, updateMask);
      classMap.setTo(c, updateMask);
    }

    cv::Mat lowConfidenceMask;
    cv::compare(maxProbs, postArgs.condThre, lowConfidenceMask, cv::CMP_LT);
    classMap.setTo(0, lowConfidenceMask);
  }

  SegRet segRet;
  segRet.clsToContours.clear();

  Shape originShape;
  const auto &inputRoi = *prepArgs.roi;
  if (inputRoi.area() > 0) {
    originShape.w = inputRoi.width;
    originShape.h = inputRoi.height;
  } else {
    originShape = prepArgs.originShape;
  }

  auto [scaleX, scaleY] = utils::scaleRatio(
      originShape, prepArgs.modelInputShape, prepArgs.isEqualScale);

  if (scaleX <= 0.0f || scaleY <= 0.0f) {
    LOG_ERRORS << "Invalid scale factors detected: scaleX=" << scaleX
               << ", scaleY=" << scaleY;
    return segRet;
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

  return segRet;
}
} // namespace ai_core::dnn
