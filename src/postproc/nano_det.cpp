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
                      const FrameTransformContext &prepArgs,
                      const AnchorDetParams &postArgs,
                      AlgoOutput &algoOutput) const {
  if (modelOutput.datas.empty()) {
    return false;
  }

  const auto &outputName = postArgs.outputNames.at(0);
  if (modelOutput.datas.find(outputName) == modelOutput.datas.end()) {
    LOG_ERRORS << "Cannot find output name " << outputName
               << " in modelOutput.";
    return false;
  }

  auto output = modelOutput.datas.at(outputName);
  std::vector<int> outputShape = modelOutput.shapes.at(outputName);

  int numAnchors = outputShape.at(outputShape.size() - 2);
  int stride = outputShape.at(outputShape.size() - 1);
  const float *outputData = output.getHostPtr<float>();

  DetRet detRet =
      processSingle(outputData, numAnchors, stride, prepArgs, postArgs);

  algoOutput.setParams(detRet);
  return true;
}

bool NanoDet::batchProcess(const TensorData &modelOutput,
                           const std::vector<FrameTransformContext> &prepArgs,
                           const AnchorDetParams &postArgs,
                           std::vector<AlgoOutput> &algoOutput) const {
  if (modelOutput.datas.empty()) {
    return false;
  }

  const auto &outputName = postArgs.outputNames.at(0);
  if (modelOutput.datas.find(outputName) == modelOutput.datas.end()) {
    LOG_ERRORS << "Cannot find output name " << outputName
               << " in modelOutput.";
    return false;
  }

  auto output = modelOutput.datas.at(outputName);
  std::vector<int> outputShape = modelOutput.shapes.at(outputName);

  if (outputShape.size() != 3) {
    LOG_ERRORS
        << "Batch process expects output tensor with 3 dimensions, but got "
        << outputShape.size();
    return false;
  }

  int batchSize = outputShape.at(0);
  int numAnchors = outputShape.at(1);
  int stride = outputShape.at(2);

  if (prepArgs.size() != batchSize) {
    LOG_ERRORS << "Batch size mismatch between model output (" << batchSize
               << ") and prepArgs (" << prepArgs.size() << ").";
    return false;
  }

  const float *allOutputData = output.getHostPtr<float>();
  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    const float *currentItemData = allOutputData + i * numAnchors * stride;
    const auto &currentItemPrepArgs = prepArgs.at(i);

    DetRet detRet = processSingle(currentItemData, numAnchors, stride,
                                  currentItemPrepArgs, postArgs);

    algoOutput[i].setParams(detRet);
  }

  return true;
}

DetRet NanoDet::processSingle(const float *outputData, int numAnchors,
                              int stride, const FrameTransformContext &prepArgs,
                              const AnchorDetParams &postArgs) const {
  cv::Mat rawData(numAnchors, stride, CV_32F, const_cast<float *>(outputData));
  int numClasses = stride - 4;

  const auto &inputRoi = *prepArgs.roi;
  Shape originShape;
  if (inputRoi.area() > 0) {
    originShape.w = inputRoi.width;
    originShape.h = inputRoi.height;
  } else {
    originShape = prepArgs.originShape;
  }
  auto [scaleX, scaleY] = utils::scaleRatio(
      originShape, prepArgs.modelInputShape, prepArgs.isEqualScale);

  std::vector<BBox> results;
  for (int i = 0; i < rawData.rows; ++i) {
    const float *data = rawData.ptr<float>(i);
    // 前 numClasses 个是分数
    cv::Mat scores(1, numClasses, CV_32F, const_cast<float *>(data));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

    if (score > postArgs.condThre) {
      BBox result;
      result.score = score;
      result.label = classIdPoint.x;

      // 接下来 4 个是坐标 (x1, y1, x2, y2)
      const float *bbox_data = data + numClasses;
      float x1 = bbox_data[0];
      float y1 = bbox_data[1];
      float x2 = bbox_data[2];
      float y2 = bbox_data[3];

      // 映射原图尺寸
      float w = (x2 - x1) / scaleX;
      float h = (y2 - y1) / scaleY;
      float x = (x1 - prepArgs.leftPad) / scaleX + inputRoi.x;
      float y = (y1 - prepArgs.topPad) / scaleY + inputRoi.y;

      result.rect =
          std::make_shared<cv::Rect>(static_cast<int>(x), static_cast<int>(y),
                                     static_cast<int>(w), static_cast<int>(h));
      results.push_back(result);
    }
  }

  DetRet detRet;
  detRet.bboxes = utils::NMS(results, postArgs.nmsThre, postArgs.condThre);
  return detRet;
}
} // namespace ai_core::dnn
