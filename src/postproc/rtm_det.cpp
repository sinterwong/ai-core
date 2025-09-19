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
bool RTMDet::process(const TensorData &modelOutput,
                     const FrameTransformContext &prepArgs,
                     const AnchorDetParams &postArgs,
                     AlgoOutput &algoOutput) const {
  if (modelOutput.datas.empty()) {
    return false;
  }

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  const auto &detOutputName = postArgs.outputNames.at(0);
  const auto &clsOutputName = postArgs.outputNames.at(1);
  auto detPred = outputs.at(detOutputName);
  auto clsPred = outputs.at(clsOutputName);

  std::vector<int> detOutShape = outputShapes.at(detOutputName);
  std::vector<int> clsOutShape = outputShapes.at(clsOutputName);

  int numClasses = clsOutShape.at(clsOutShape.size() - 1);
  int anchorNum = detOutShape.at(detOutShape.size() - 2);

  DetRet detRet =
      processSingle(detPred.getHostPtr<float>(), clsPred.getHostPtr<float>(),
                    anchorNum, numClasses, prepArgs, postArgs);

  algoOutput.setParams(detRet);
  return true;
}

bool RTMDet::batchProcess(const TensorData &modelOutput,
                          const std::vector<FrameTransformContext> &prepArgs,
                          const AnchorDetParams &postArgs,
                          std::vector<AlgoOutput> &algoOutput) const {
  if (modelOutput.datas.empty()) {
    return false;
  }

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  const auto &detOutputName = postArgs.outputNames.at(0);
  const auto &clsOutputName = postArgs.outputNames.at(1);
  auto detPred = outputs.at(detOutputName);
  auto clsPred = outputs.at(clsOutputName);

  std::vector<int> detOutShape = outputShapes.at(detOutputName);
  std::vector<int> clsOutShape = outputShapes.at(clsOutputName);

  int batchSize = detOutShape.at(0);
  int anchorNum = detOutShape.at(1);
  int numClasses = clsOutShape.at(2);

  if (prepArgs.size() != batchSize) {
    LOG_ERRORS << "Batch size mismatch between model output (" << batchSize
               << ") and prepArgs (" << prepArgs.size() << ").";
    return false;
  }

  const float *detDataPtr = detPred.getHostPtr<float>();
  const float *clsDataPtr = clsPred.getHostPtr<float>();

  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    const float *currentDetData = detDataPtr + i * anchorNum * 4;
    const float *currentClsData = clsDataPtr + i * anchorNum * numClasses;
    const auto &currentPrepArgs = prepArgs[i];

    DetRet detRet = processSingle(currentDetData, currentClsData, anchorNum,
                                  numClasses, currentPrepArgs, postArgs);

    algoOutput[i].setParams(detRet);
  }
  return true;
}

DetRet RTMDet::processSingle(const float *detDataPtr, const float *clsDataPtr,
                             int anchorNum, int numClasses,
                             const FrameTransformContext &prepArgs,
                             const AnchorDetParams &postArgs) const {
  const auto &inputShape = prepArgs.modelInputShape;
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

  std::vector<BBox> results;
  for (int i = 0; i < anchorNum; ++i) {
    auto detData = detDataPtr + i * 4;
    auto clsData = clsDataPtr + i * numClasses;
    cv::Mat scores(1, numClasses, CV_32F, const_cast<float *>(clsData));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);
    if (score > postArgs.condThre) {
      float x = detData[0];
      float y = detData[1];
      float w = detData[2] - x;
      float h = detData[3] - y;

      if (prepArgs.isEqualScale) {
        x = (x - prepArgs.leftPad) / scaleX;
        y = (y - prepArgs.topPad) / scaleY;
      } else {
        x = x / scaleX;
        y = y / scaleY;
      }
      w = w / scaleX;
      h = h / scaleY;

      BBox result;
      result.score = score;
      result.label = classIdPoint.x;
      x += inputRoi.x;
      y += inputRoi.y;

      result.rect =
          std::make_shared<cv::Rect>(static_cast<int>(x), static_cast<int>(y),
                                     static_cast<int>(w), static_cast<int>(h));
      results.emplace_back(result);
    }
  }

  DetRet detRet;
  detRet.bboxes = utils::NMS(results, postArgs.nmsThre, postArgs.condThre);
  return detRet;
}

} // namespace ai_core::dnn
