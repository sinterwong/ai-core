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
#include "ai_core/output_types.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool OCRReco::processTyped(const TensorData &model_output,
                      const FrameTransformContext &prep_args,
                      const GenericPostParams &post_args,
                      AlgoOutput &algo_output) const {
  const auto &output_lengths_name = post_args.output_names.at(0);
  const auto &argmax_output_name = post_args.output_names.at(1);

  const auto &outputs = model_output.datas;
  auto p_output_lengths = outputs.at(output_lengths_name);
  auto p_argmax_outputs = outputs.at(argmax_output_name);

  const int64_t *length_data = p_output_lengths.getHostPtr<int64_t>();
  const int64_t *argmax_data = p_argmax_outputs.getHostPtr<int64_t>();
  const size_t sequence_length = p_argmax_outputs.getElementCount();

  OCRRecoRet ocr_ret =
      processSingleItem(argmax_data, sequence_length, *length_data);

  algo_output.setParams(ocr_ret);
  return true;
}

bool OCRReco::batchProcessTyped(const TensorData &model_output,
                           const std::vector<FrameTransformContext> &prep_args,
                           const GenericPostParams &post_args,
                           std::vector<AlgoOutput> &algo_output) const {
  const auto &output_lengths_name = post_args.output_names.at(0);
  const auto &argmax_output_name = post_args.output_names.at(1);

  const auto &output_shapes = model_output.shapes;
  const auto &outputs = model_output.datas;

  auto p_output_lengths = outputs.at(output_lengths_name);
  auto p_argmax_outputs = outputs.at(argmax_output_name);

  const std::vector<int> &argmax_shape = output_shapes.at(argmax_output_name);
  if (argmax_shape.size() != 2) {
    // Handle error: shape is not as expected for a batch
    return false;
  }
  const int batch_size = argmax_shape.at(0);
  const int sequence_length = argmax_shape.at(1);

  const int64_t *lengths_data = p_output_lengths.getHostPtr<int64_t>();
  const int64_t *argmax_data = p_argmax_outputs.getHostPtr<int64_t>();

  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    // 计算当前样本在内存块中的起始位置
    const int64_t *current_argmax_data =
        argmax_data + (size_t)i * sequence_length;
    const int64_t current_output_length = lengths_data[i];

    OCRRecoRet ocr_ret = processSingleItem(current_argmax_data, sequence_length,
                                           current_output_length);

    algo_output[i].setParams(ocr_ret);
  }

  return true;
}

OCRRecoRet OCRReco::processSingleItem(const int64_t *argmax_data,
                                      size_t sequence_length,
                                      int64_t output_length) const {
  OCRRecoRet ocr_ret;
  ocr_ret.output_lengths = output_length;
  std::vector<int64_t> argmax_outputs_vec(argmax_data,
                                          argmax_data + sequence_length);
  ocr_ret.outputs = ctcProcess(argmax_outputs_vec);
  return ocr_ret;
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
