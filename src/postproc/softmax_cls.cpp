/**
 * @file softmax_cls.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "softmax_cls.hpp"
#include "logger.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool SoftmaxCls::process(const TensorData &modelOutput,
                         const FrameTransformContext &prepArgs,
                         const GenericPostParams &postArgs,
                         AlgoOutput &algoOutput) const {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.datas is empty";
    return false;
  }

  const auto &scoreOutputName = postArgs.outputNames.at(0);
  const auto &output = modelOutput.datas.at(scoreOutputName);
  const auto &outputShape = modelOutput.shapes.at(scoreOutputName);

  int numClasses = outputShape.at(outputShape.size() - 1);

  const float *logits = output.getHostPtr<float>();

  ClsRet clsRet = processSingleItem(logits, numClasses);

  algoOutput.setParams(clsRet);
  return true;
}

bool SoftmaxCls::batchProcess(
    const TensorData &modelOutput,
    const std::vector<FrameTransformContext> &prepArgs,
    const GenericPostParams &postArgs,
    std::vector<AlgoOutput> &algoOutput) const {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.datas is empty";
    return false;
  }

  const auto &scoreOutputName = postArgs.outputNames.at(0);
  const auto &output = modelOutput.datas.at(scoreOutputName);
  const auto &outputShape = modelOutput.shapes.at(scoreOutputName);

  if (outputShape.size() != 2) {
    LOG_ERRORS
        << "Expected a 2D tensor for batch classification (N, C), but got "
        << outputShape.size() << " dimensions.";
    return false;
  }

  const int batchSize = outputShape.at(0);
  const int numClasses = outputShape.at(1);

  const float *baseLogits = output.getHostPtr<float>();

  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    const float *currentLogits = baseLogits + i * numClasses;
    ClsRet clsRet = processSingleItem(currentLogits, numClasses);
    algoOutput[i].setParams(clsRet);
  }
  return true;
}

ClsRet SoftmaxCls::processSingleItem(const float *logits,
                                     int numClasses) const {
  cv::Mat logitMat(1, numClasses, CV_32F, const_cast<float *>(logits));

  double maxLogit;
  cv::minMaxLoc(logitMat, nullptr, &maxLogit, nullptr, nullptr);

  cv::Mat expMat;
  cv::exp(logitMat - maxLogit, expMat);

  double sum = cv::sum(expMat)[0];

  cv::Mat probMat = expMat / sum;

  cv::Point classIdPoint;
  double score;
  cv::minMaxLoc(probMat, nullptr, &score, nullptr, &classIdPoint);

  ClsRet clsRet;
  clsRet.score = static_cast<float>(score);
  clsRet.label = classIdPoint.x;

  return clsRet;
}
} // namespace ai_core::dnn
