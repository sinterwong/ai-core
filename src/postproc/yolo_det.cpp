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
#include "yolo_det.hpp"
#include "logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool Yolov11Det::process(const TensorData &modelOutput,
                         const FrameTransformContext &prepArgs,
                         const AnchorDetParams &postArgs,
                         AlgoOutput &algoOutput) const {
  const auto &outputShapes = modelOutput.shapes;
  const auto &inputShape = prepArgs.modelInputShape;
  const auto &outputs = modelOutput.datas;

  // just one output
  if (outputs.size() != 1) {
    LOG_ERRORS << "AnchorDetParams(Yolov11Det) unexpected size of outputs "
               << outputs.size();
    throw std::runtime_error(
        "AnchorDetParams(Yolov11Det)  unexpected size of outputs");
  }
  auto output = outputs.at(postArgs.outputNames.at(0));

  std::vector<int> outputShape = outputShapes.at(postArgs.outputNames.at(0));
  int signalResultNum = outputShape.at(outputShape.size() - 2);
  int strideNum = outputShape.at(outputShape.size() - 1);

  cv::Mat rawData = cv::Mat(strideNum, signalResultNum, CV_32F);
  if (output.dataType() == DataType::FLOAT32) {
    cv::transpose(cv::Mat(signalResultNum, strideNum, CV_32F,
                          const_cast<void *>(output.getRawHostPtr())),
                  rawData);
  } else if (output.dataType() == DataType::FLOAT16) {
    const uint16_t *fp16Data = output.getHostPtr<uint16_t>();
    cv::Mat halfMat(1, output.getElementCount(), CV_16F, (void *)fp16Data);
    cv::Mat floatMat(1, output.getElementCount(), CV_32F);
    halfMat.convertTo(floatMat, CV_32F);
    cv::transpose(cv::Mat(signalResultNum, strideNum, CV_32F, floatMat.data),
                  rawData);
  }

  std::vector<BBox> results = processRawOutput(rawData, inputShape, prepArgs,
                                               postArgs, signalResultNum - 4);

  DetRet detRet;
  detRet.bboxes = utils::NMS(results, postArgs.nmsThre, postArgs.condThre);
  algoOutput.setParams(detRet);
  return true;
}

std::vector<BBox> Yolov11Det::processRawOutput(
    const cv::Mat &transposedOutput, const Shape &inputShape,
    const FrameTransformContext &prepArgs, const AnchorDetParams &postArgs,
    int numClasses) const {
  std::vector<BBox> results;
  Shape originShape;
  const auto &inputRoi = *prepArgs.roi;
  if (inputRoi.area() > 0) {
    originShape.w = inputRoi.width;
    originShape.h = inputRoi.height;
  } else {
    originShape = prepArgs.originShape;
  }
  auto [scaleX, scaleY] =
      utils::scaleRatio(originShape, inputShape, prepArgs.isEqualScale);

  for (int i = 0; i < transposedOutput.rows; ++i) {
    const float *data = transposedOutput.ptr<float>(i);

    cv::Mat scores(1, numClasses, CV_32F, (void *)(data + 4));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

    if (score > postArgs.condThre) {
      BBox result;
      result.score = score;
      result.label = classIdPoint.x;

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      x = x - 0.5 * w;
      y = y - 0.5 * h;

      if (prepArgs.isEqualScale) {
        x = (x - prepArgs.leftPad) / scaleX;
        y = (y - prepArgs.topPad) / scaleY;
      } else {
        x = x / scaleX;
        y = y / scaleY;
      }
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

  return results;
}
} // namespace ai_core::dnn
