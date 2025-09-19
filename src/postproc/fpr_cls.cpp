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

  const auto &outputs = modelOutput.datas;
  auto pScores = outputs.at(scoreOutputName);
  auto pBirads = outputs.at(biradOutputName);

  std::vector<int> pScoresShape = modelOutput.shapes.at(scoreOutputName);
  int numClasses = pScoresShape.at(pScoresShape.size() - 1);

  std::vector<int> pBiradsShape = modelOutput.shapes.at(biradOutputName);
  int numBirads = pBiradsShape.at(pBiradsShape.size() - 1);

  FprClsRet fprRet = processSingleItem(pScores.getHostPtr<float>(), numClasses,
                                       pBirads.getHostPtr<float>(), numBirads);

  algoOutput.setParams(fprRet);
  return true;
}

bool FprCls::batchProcess(const TensorData &modelOutput,
                          const std::vector<FrameTransformContext> &prepArgs,
                          const GenericPostParams &postArgs,
                          std::vector<AlgoOutput> &algoOutput) const {
  const auto &scoreOutputName = postArgs.outputNames.at(0);
  const auto &biradOutputName = postArgs.outputNames.at(1);

  const auto &outputs = modelOutput.datas;
  auto pScores = outputs.at(scoreOutputName);
  auto pBirads = outputs.at(biradOutputName);

  std::vector<int> pScoresShape = modelOutput.shapes.at(scoreOutputName);
  int batchSize = pScoresShape.at(0);
  int numClasses = pScoresShape.at(pScoresShape.size() - 1);

  std::vector<int> pBiradsShape = modelOutput.shapes.at(biradOutputName);
  int numBirads = pBiradsShape.at(pBiradsShape.size() - 1);

  const float *scoresData = pScores.getHostPtr<float>();
  const float *biradsData = pBirads.getHostPtr<float>();

  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    const float *currentScores = scoresData + i * numClasses;
    const float *currentBirads = biradsData + i * numBirads;

    FprClsRet fprRet =
        processSingleItem(currentScores, numClasses, currentBirads, numBirads);
    algoOutput[i].setParams(fprRet);
  }
  return true;
}

FprClsRet FprCls::processSingleItem(const float *scoresData, int numClasses,
                                    const float *biradsData,
                                    int numBirads) const {
  cv::Mat scores(1, numClasses, CV_32F, const_cast<float *>(scoresData));
  cv::Mat birads(1, numBirads, CV_32F, const_cast<float *>(biradsData));

  cv::Point classIdPoint;
  double score;
  cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

  cv::Point biradsIdPoint;
  double biradsScore;
  cv::minMaxLoc(birads, nullptr, &biradsScore, nullptr, &biradsIdPoint);

  FprClsRet fprRet;
  fprRet.score = score;
  fprRet.label = classIdPoint.x;
  fprRet.scoreProbs.assign(scoresData, scoresData + numClasses);
  fprRet.birad = biradsIdPoint.x;

  return fprRet;
}
} // namespace ai_core::dnn
