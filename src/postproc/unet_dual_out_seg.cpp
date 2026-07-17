/**
 * @file unet_dual_out_seg.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "unet_dual_out_seg.hpp"

#include "ai_core/logger.hpp"

#include <cstring>

namespace ai_core::dnn {
bool UNetDualOutputSeg::processTyped(const TensorData &model_output,
                                     const FrameTransformContext &prep_args,
                                     const GenericPostParams &post_args,
                                     AlgoOutput &algo_output) const {
  if (post_args.output_names.size() != 2) {
    LOG_ERROR_S
        << "UNetDualOutputSeg expects exactly two output names: prob and mask.";
    return false;
  }
  const auto &prob_output_name = post_args.output_names.at(0);
  const auto &mask_output_name = post_args.output_names.at(1);

  const auto &prob_output = model_output.at(prob_output_name).buffer;
  const auto &mask_output = model_output.at(mask_output_name).buffer;
  const auto &prob_shape = model_output.at(prob_output_name).shape;
  const auto &mask_shape = model_output.at(mask_output_name).shape;

  DualRawSegRet ret =
      processSingleItem(prob_output.getHostPtr<float>(), prob_shape,
                        mask_output.getHostPtr<float>(), mask_shape, prep_args);

  algo_output.setParams(ret);
  return true;
}

bool UNetDualOutputSeg::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const GenericPostParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {

  if (post_args.output_names.size() != 2) {
    LOG_ERROR_S
        << "UNetDualOutputSeg expects exactly two output names: prob and mask.";
    return false;
  }
  const auto &prob_output_name = post_args.output_names.at(0);
  const auto &mask_output_name = post_args.output_names.at(1);

  const auto &prob_output = model_output.at(prob_output_name).buffer;
  const auto &mask_output = model_output.at(mask_output_name).buffer;
  const auto &prob_shape = model_output.at(prob_output_name).shape;
  const auto &mask_shape = model_output.at(mask_output_name).shape;

  int batch_size = prob_shape.at(0);
  if (batch_size != prep_args.size()) {
    LOG_ERROR_S << "Batch size from model output (" << batch_size
                << ") does not match prep_args size (" << prep_args.size()
                << ").";
    return false;
  }

  // 计算单个样本的元素数量
  size_t prob_item_size = prob_output.getElementCount() / batch_size;
  size_t mask_item_size = mask_output.getElementCount() / batch_size;

  const float *prob_data_ptr = prob_output.getHostPtr<float>();
  const float *mask_data_ptr = mask_output.getHostPtr<float>();

  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    const float *current_prob_data = prob_data_ptr + i * prob_item_size;
    const float *current_mask_data = mask_data_ptr + i * mask_item_size;

    // 在循环中调用辅助函数
    DualRawSegRet ret =
        processSingleItem(current_prob_data, prob_shape, current_mask_data,
                          mask_shape, prep_args[i]);
    algo_output[i].setParams(ret);
  }

  return true;
}

DualRawSegRet UNetDualOutputSeg::processSingleItem(
    const float *prob_data, const std::vector<int> &prob_shape,
    const float *mask_data, const std::vector<int> &mask_shape,
    const FrameTransformContext &prep_args) const {

  // The result owns its pixels: copy out of the inference output buffer so
  // the returned tensors stay valid after the TensorData is released.
  auto copyPlane = [](const char *tag, const float *data,
                      const std::vector<int> &shape) {
    const int height = shape[2];
    const int width = shape[1];
    const size_t byte_size =
        static_cast<size_t>(height) * width * sizeof(float);
    std::vector<uint8_t> bytes(byte_size);
    std::memcpy(bytes.data(), data, byte_size);
    return Tensor{tag,
                  TypedBuffer::createFromCpu(DataType::FLOAT32,
                                             std::move(bytes)),
                  {height, width}};
  };

  DualRawSegRet ret;
  ret.prob = copyPlane("prob", prob_data, prob_shape);
  ret.mask = copyPlane("mask", mask_data, mask_shape);

  ret.roi = prep_args.roi;
  ret.ratio = static_cast<float>(prep_args.model_input_shape.w) /
              prep_args.origin_shape.w;
  ret.left_shift = prep_args.left_pad;
  ret.top_shift = prep_args.top_pad;
  return ret;
}
} // namespace ai_core::dnn
