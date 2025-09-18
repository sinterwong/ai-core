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

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  DaulRawSegRet ret;

  auto probOutput = outputs.at(probOutputName);
  auto maskOutput = outputs.at(maskOutputName);

  std::vector<int> probShape = outputShapes.at(probOutputName);
  std::vector<int> maskShape = outputShapes.at(maskOutputName);

  cv::Mat probCvMat(probShape[2], probShape[1], CV_32FC1,
                    const_cast<void *>(probOutput.getRawHostPtr()));
  cv::Mat maskCvMat(maskShape[2], maskShape[1], CV_32FC1,
                    const_cast<void *>(maskOutput.getRawHostPtr()));

  ret.prob = std::make_shared<cv::Mat>(probCvMat);
  ret.mask = std::make_shared<cv::Mat>(maskCvMat);

  ret.roi = prepArgs.roi;
  ret.ratio =
      static_cast<float>(prepArgs.modelInputShape.w) / prepArgs.originShape.w;
  ret.leftShift = prepArgs.leftPad;
  ret.topShift = prepArgs.topPad;

  algoOutput.setParams(ret);
  return true;
}
} // namespace ai_core::dnn
