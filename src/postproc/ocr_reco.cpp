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
#include "ocr_reco.hpp"
#include "ai_core/algo_output_types.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool OCRReco::process(const TensorData &modelOutput,
                      const FrameTransformContext &prepArgs,
                      const GenericPostParams &postArgs,
                      AlgoOutput &algoOutput) const {
  const auto &outputLengthsName = postArgs.outputNames.at(0);
  const auto &argmaxOutputName = postArgs.outputNames.at(1);

  const auto &outputs = modelOutput.datas;
  auto pOutputLengths = outputs.at(outputLengthsName);
  auto pArgmaxOutputs = outputs.at(argmaxOutputName);

  const int64_t *lengthData = pOutputLengths.getHostPtr<int64_t>();
  const int64_t *argmaxData = pArgmaxOutputs.getHostPtr<int64_t>();
  const size_t sequenceLength = pArgmaxOutputs.getElementCount();

  OCRRecoRet ocrRet =
      processSingleItem(argmaxData, sequenceLength, *lengthData);

  algoOutput.setParams(ocrRet);
  return true;
}

bool OCRReco::batchProcess(const TensorData &modelOutput,
                           const std::vector<FrameTransformContext> &prepArgs,
                           const GenericPostParams &postArgs,
                           std::vector<AlgoOutput> &algoOutput) const {
  const auto &outputLengthsName = postArgs.outputNames.at(0);
  const auto &argmaxOutputName = postArgs.outputNames.at(1);

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  auto pOutputLengths = outputs.at(outputLengthsName);
  auto pArgmaxOutputs = outputs.at(argmaxOutputName);

  const std::vector<int> &argmaxShape = outputShapes.at(argmaxOutputName);
  if (argmaxShape.size() != 2) {
    // Handle error: shape is not as expected for a batch
    return false;
  }
  const int batchSize = argmaxShape.at(0);
  const int sequenceLength = argmaxShape.at(1);

  const int64_t *lengthsData = pOutputLengths.getHostPtr<int64_t>();
  const int64_t *argmaxData = pArgmaxOutputs.getHostPtr<int64_t>();

  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    // 计算当前样本在内存块中的起始位置
    const int64_t *currentArgmaxData = argmaxData + (size_t)i * sequenceLength;
    const int64_t currentOutputLength = lengthsData[i];

    OCRRecoRet ocrRet = processSingleItem(currentArgmaxData, sequenceLength,
                                          currentOutputLength);

    algoOutput[i].setParams(ocrRet);
  }

  return true;
}

OCRRecoRet OCRReco::processSingleItem(const int64_t *argmaxData,
                                      size_t sequenceLength,
                                      int64_t outputLength) const {
  OCRRecoRet ocrRet;
  ocrRet.outputLengths = outputLength;
  std::vector<int64_t> argmaxOutputsVec(argmaxData,
                                        argmaxData + sequenceLength);
  ocrRet.outputs = ctcProcess(argmaxOutputsVec);
  return ocrRet;
}

std::vector<int64_t>
OCRReco::ctcProcess(const std::vector<int64_t> &outputs) const {
  std::vector<int64_t> result;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] != 0 && !(i > 0 && outputs[i] == outputs[i - 1])) {
      result.push_back(outputs[i]);
    }
  }
  return result;
}
} // namespace ai_core::dnn
