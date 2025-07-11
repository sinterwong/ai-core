/**
 * @file rtmDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "rtm_det.hpp"
#include "logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool RTMDet::process(const TensorData &modelOutput, AlgoPreprocParams &prepArgs,
                     AlgoOutput &algoOutput, AlgoPostprocParams &postArgs) {
  if (modelOutput.datas.empty()) {
    return false;
  }

  const auto &prepParams = prepArgs.getParams<FramePreprocessArg>();
  if (prepParams == nullptr) {
    LOG_ERRORS << "FramePreprocessArg is nullptr";
    throw std::runtime_error("FramePreprocessArg is nullptr");
  }

  auto params = postArgs.getParams<AnchorDetParams>();
  if (params == nullptr) {
    LOG_ERRORS << "AnchorDetParams params is nullptr";
    throw std::runtime_error("AnchorDetParams params is nullptr");
  }

  const auto &outputShapes = modelOutput.shapes;
  const auto &inputShape = prepParams->modelInputShape;
  const auto &outputs = modelOutput.datas;

  // two output
  const auto &detOutputName = params->outputNames.at(0);
  const auto &clsOutputName = params->outputNames.at(1);
  auto detPred = outputs.at(detOutputName);
  auto clsPred = outputs.at(clsOutputName);

  std::vector<int> detOutShape = outputShapes.at(detOutputName);
  std::vector<int> clsOutShape = outputShapes.at(clsOutputName);

  int numClasses = clsOutShape.at(clsOutShape.size() - 1);
  int anchorNum = detOutShape.at(detOutShape.size() - 2);

  Shape originShape;
  if (prepParams->roi.area() > 0) {
    originShape.w = prepParams->roi.width;
    originShape.h = prepParams->roi.height;
  } else {
    originShape = prepParams->originShape;
  }
  auto [scaleX, scaleY] =
      utils::scaleRatio(originShape, inputShape, prepParams->isEqualScale);

  std::vector<BBox> results;
  auto detDataPtr = detPred.getHostPtr<float>();
  auto clsDataPtr = clsPred.getHostPtr<float>();
  for (int i = 0; i < anchorNum; ++i) {
    auto detData = detDataPtr + i * 4;
    auto clsData = clsDataPtr + i * numClasses;
    cv::Mat scores(1, numClasses, CV_32F, const_cast<float *>(clsData));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);
    if (score > params->condThre) {
      float x = detData[0];
      float y = detData[1];
      float w = detData[2] - x;
      float h = detData[3] - y;

      if (prepParams->isEqualScale) {
        x = (x - prepParams->leftPad) / scaleX;
        y = (y - prepParams->topPad) / scaleY;
      } else {
        x = x / scaleX;
        y = y / scaleY;
      }
      w = w / scaleX;
      h = h / scaleY;
      BBox result;
      result.score = score;
      result.label = classIdPoint.x;
      x += prepParams->roi.x;
      y += prepParams->roi.y;

      result.rect = {static_cast<int>(x), static_cast<int>(y),
                     static_cast<int>(w), static_cast<int>(h)};
      results.emplace_back(result);
    }
  }

  DetRet detRet;
  detRet.bboxes = utils::NMS(results, params->nmsThre, params->condThre);
  algoOutput.setParams(detRet);
  return true;
}

} // namespace ai_core::dnn
