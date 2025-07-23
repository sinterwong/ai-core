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
#include "vision_util.hpp"
#include <logger.hpp>
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool NanoDet::process(const TensorData &modelOutput,
                      const FramePreprocessArg &prepArgs,
                      AlgoOutput &algoOutput,
                      const AnchorDetParams &postArgs) const {
  if (modelOutput.datas.empty()) {
    return false;
  }

  const auto &outputShapes = modelOutput.shapes;
  const auto &inputShape = prepArgs.modelInputShape;
  const auto &outputs = modelOutput.datas;

  // just one output
  if (outputs.size() != 1) {
    LOG_ERRORS << "AnchorDetParams(NanoDet) unexpected size of outputs "
               << outputs.size();
    throw std::runtime_error(
        "AnchorDetParams(NanoDet)  unexpected size of outputs");
  }
  auto output = outputs.at(postArgs.outputNames.at(0));

  std::vector<int> outputShape = outputShapes.at(postArgs.outputNames.at(0));
  int numAnchors = outputShape.at(outputShape.size() - 2);
  int stride = outputShape.at(outputShape.size() - 1);
  int numClasses = stride - 4;

  // [1, 3598, 11]
  cv::Mat rawData(numAnchors, stride, CV_32F,
                  const_cast<void *>(output.getRawHostPtr()));

  const auto &inputRoi = *prepArgs.roi;

  Shape originShape;
  if (inputRoi.area() > 0) {
    originShape.w = inputRoi.width;
    originShape.h = inputRoi.height;
  } else {
    originShape = prepArgs.originShape;
  }
  auto [scaleX, scaleY] =
      utils::scaleRatio(originShape, inputShape, prepArgs.isEqualScale);

  std::vector<BBox> results;
  for (int i = 0; i < rawData.rows; ++i) {
    const float *data =
        rawData.ptr<float>(i); // Now data points to valid output data
    cv::Mat scores(1, numClasses, CV_32F, (void *)(data));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

    if (score > postArgs.condThre) {
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

      x = (x1 - prepArgs.leftPad) / scaleX;
      y = (y1 - prepArgs.topPad) / scaleY;
      w = w / scaleX;
      h = h / scaleY;
      x += inputRoi.x;
      y += inputRoi.y;
      result.rect =
          std::make_shared<cv::Rect>(static_cast<int>(x), static_cast<int>(y),
                                     static_cast<int>(w), static_cast<int>(h));
      results.push_back(result);
    }
  }
  DetRet detRet;
  detRet.bboxes = utils::NMS(results, postArgs.nmsThre, postArgs.condThre);
  algoOutput.setParams(detRet);
  return true;
}
} // namespace ai_core::dnn
