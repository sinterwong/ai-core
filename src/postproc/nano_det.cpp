/**
 * @file yoloDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "nano_det.hpp"
#include "logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool NanoDet::process(const TensorData &modelOutput,
                      AlgoPreprocParams &prepArgs, AlgoOutput &algoOutput,
                      AlgoPostprocParams &postArgs) {
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

  // just one output
  if (outputs.size() != 1) {
    LOG_ERRORS << "AnchorDetParams(NanoDet) unexpected size of outputs "
               << outputs.size();
    throw std::runtime_error(
        "AnchorDetParams(NanoDet)  unexpected size of outputs");
  }
  auto output = outputs.at(params->outputNames.at(0));

  std::vector<int> outputShape = outputShapes.at(params->outputNames.at(0));
  int numAnchors = outputShape.at(outputShape.size() - 2);
  int stride = outputShape.at(outputShape.size() - 1);
  int numClasses = stride - 4;

  // [1, 3598, 11]
  cv::Mat rawData(numAnchors, stride, CV_32F,
                  const_cast<void *>(output.getRawHostPtr()));

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
  for (int i = 0; i < rawData.rows; ++i) {
    const float *data =
        rawData.ptr<float>(i); // Now data points to valid output data
    cv::Mat scores(1, numClasses, CV_32F, (void *)(data));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

    if (score > params->condThre) {
      BBox result;
      result.score = score;
      result.label = classIdPoint.x;

      float x1 = data[numClasses + 0];
      float y1 = data[numClasses + 1];
      float x2 = data[numClasses + 2];
      float y2 = data[numClasses + 3];
      float w = x2 - x1;
      float h = y2 - y1;
      float x, y;

      x = (x1 - prepParams->leftPad) / scaleX;
      y = (y1 - prepParams->topPad) / scaleY;
      w = w / scaleX;
      h = h / scaleY;
      x += prepParams->roi.x;
      y += prepParams->roi.y;
      result.rect = {static_cast<int>(x), static_cast<int>(y),
                     static_cast<int>(w), static_cast<int>(h)};
      results.push_back(result);
    }
  }
  DetRet detRet;
  detRet.bboxes = utils::NMS(results, params->nmsThre, params->condThre);
  algoOutput.setParams(detRet);
  return true;
}
} // namespace ai_core::dnn
