/**
 * @file unet_daul_out_seg.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "unet_daul_out_seg.hpp"

#include <logger.hpp>
#include <opencv2/core/mat.hpp>

namespace ai_core::dnn {
bool UNetDaulOutputSeg::process(const TensorData &modelOutput,
                                const FrameTransformContext &prepArgs,
                                const GenericPostParams &postArgs,
                                AlgoOutput &algoOutput) const {
  if (postArgs.outputNames.size() != 2) {
    LOG_ERRORS
        << "UNetDaulOutputSeg expects exactly two output names: prob and mask.";
    return false;
  }
  const auto &probOutputName = postArgs.outputNames.at(0);
  const auto &maskOutputName = postArgs.outputNames.at(1);

  auto probOutput = modelOutput.datas.at(probOutputName);
  auto maskOutput = modelOutput.datas.at(maskOutputName);
  const auto &probShape = modelOutput.shapes.at(probOutputName);
  const auto &maskShape = modelOutput.shapes.at(maskOutputName);

  DaulRawSegRet ret =
      processSingleItem(probOutput.getHostPtr<float>(), probShape,
                        maskOutput.getHostPtr<float>(), maskShape, prepArgs);

  algoOutput.setParams(ret);
  return true;
}

bool UNetDaulOutputSeg::batchProcess(
    const TensorData &modelOutput,
    const std::vector<FrameTransformContext> &prepArgs,
    const GenericPostParams &postArgs,
    std::vector<AlgoOutput> &algoOutput) const {

  if (postArgs.outputNames.size() != 2) {
    LOG_ERRORS
        << "UNetDaulOutputSeg expects exactly two output names: prob and mask.";
    return false;
  }
  const auto &probOutputName = postArgs.outputNames.at(0);
  const auto &maskOutputName = postArgs.outputNames.at(1);

  auto probOutput = modelOutput.datas.at(probOutputName);
  auto maskOutput = modelOutput.datas.at(maskOutputName);
  const auto &probShape = modelOutput.shapes.at(probOutputName);
  const auto &maskShape = modelOutput.shapes.at(maskOutputName);

  int batchSize = probShape.at(0);
  if (batchSize != prepArgs.size()) {
    LOG_ERRORS << "Batch size from model output (" << batchSize
               << ") does not match prepArgs size (" << prepArgs.size() << ").";
    return false;
  }

  // 计算单个样本的元素数量
  size_t probItemSize = probOutput.getElementCount() / batchSize;
  size_t maskItemSize = maskOutput.getElementCount() / batchSize;

  const float *probDataPtr = probOutput.getHostPtr<float>();
  const float *maskDataPtr = maskOutput.getHostPtr<float>();

  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    const float *currentProbData = probDataPtr + i * probItemSize;
    const float *currentMaskData = maskDataPtr + i * maskItemSize;

    // 在循环中调用辅助函数
    DaulRawSegRet ret = processSingleItem(
        currentProbData, probShape, currentMaskData, maskShape, prepArgs[i]);
    algoOutput[i].setParams(ret);
  }

  return true;
}

DaulRawSegRet UNetDaulOutputSeg::processSingleItem(
    const float *probData, const std::vector<int> &probShape,
    const float *maskData, const std::vector<int> &maskShape,
    const FrameTransformContext &prepArgs) const {

  int height = probShape[2];
  int width = probShape[1];
  cv::Mat probCvMat(height, width, CV_32FC1, const_cast<float *>(probData));

  height = maskShape[2];
  width = maskShape[1];
  cv::Mat maskCvMat(height, width, CV_32FC1, const_cast<float *>(maskData));

  DaulRawSegRet ret;
  ret.prob = std::make_shared<cv::Mat>(probCvMat);
  ret.mask = std::make_shared<cv::Mat>(maskCvMat);

  ret.roi = prepArgs.roi;
  ret.ratio =
      static_cast<float>(prepArgs.modelInputShape.w) / prepArgs.originShape.w;
  ret.leftShift = prepArgs.leftPad;
  ret.topShift = prepArgs.topPad;
  return ret;
}
} // namespace ai_core::dnn
