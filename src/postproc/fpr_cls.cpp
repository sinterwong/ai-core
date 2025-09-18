/**
 * @file fpr_cls.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "fpr_cls.hpp"
#include "ai_core/algo_output_types.hpp"

#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool FprCls::process(const TensorData &modelOutput,
                     const FrameTransformContext &prepArgs,
                     const GenericPostParams &postArgs,
                     AlgoOutput &algoOutput) const {
  const auto &scoreOutputName = postArgs.outputNames.at(0);
  const auto &biradOutputName = postArgs.outputNames.at(1);

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  auto pScores = outputs.at(scoreOutputName);
  auto pBirads = outputs.at(biradOutputName);

  std::vector<int> pScoresShape = outputShapes.at(scoreOutputName);
  int numClasses = pScoresShape.at(pScoresShape.size() - 1);

  std::vector<int> pBiradsShape = outputShapes.at(biradOutputName);
  int numBirads = pBiradsShape.at(pBiradsShape.size() - 1);

  cv::Mat scores(1, numClasses, CV_32F, pScores.getRawHostPtr());
  cv::Mat birads(1, numBirads, CV_32F, pBirads.getRawHostPtr());

  cv::Point classIdPoint;
  double score;
  cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

  cv::Point biradsIdPoint;
  double biradsScore;
  cv::minMaxLoc(birads, nullptr, &biradsScore, nullptr, &biradsIdPoint);

  FprClsRet fprRet;
  fprRet.score = score;
  fprRet.label = classIdPoint.x;
  fprRet.scoreProbs.assign(pScores.getHostPtr<float>(),
                           pScores.getHostPtr<float>() +
                               pScores.getElementCount());
  fprRet.birad = biradsIdPoint.x;
  algoOutput.setParams(fprRet);
  return true;
}
} // namespace ai_core::dnn
