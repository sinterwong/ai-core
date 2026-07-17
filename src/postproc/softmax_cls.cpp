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
#include "ai_core/logger.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool SoftmaxCls::processTyped(const TensorData &model_output,
                              const FrameTransformContext &prep_args,
                              const GenericPostParams &post_args,
                              AlgoOutput &algo_output) const {
  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.datas is empty";
    return false;
  }

  const auto &score_output_name = post_args.output_names.at(0);
  const auto &output = model_output.datas.at(score_output_name);
  const auto &output_shape = model_output.shapes.at(score_output_name);

  int num_classes = output_shape.at(output_shape.size() - 1);

  const float *logits = output.getHostPtr<float>();

  ClsRet cls_ret = processSingleItem(logits, num_classes);

  algo_output.setParams(cls_ret);
  return true;
}

bool SoftmaxCls::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const GenericPostParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {
  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.datas is empty";
    return false;
  }

  const auto &score_output_name = post_args.output_names.at(0);
  const auto &output = model_output.datas.at(score_output_name);
  const auto &output_shape = model_output.shapes.at(score_output_name);

  if (output_shape.size() != 2) {
    LOG_ERROR_S
        << "Expected a 2D tensor for batch classification (N, C), but got "
        << output_shape.size() << " dimensions.";
    return false;
  }

  const int batch_size = output_shape.at(0);
  const int num_classes = output_shape.at(1);

  const float *base_logits = output.getHostPtr<float>();

  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    const float *current_logits = base_logits + i * num_classes;
    ClsRet cls_ret = processSingleItem(current_logits, num_classes);
    algo_output[i].setParams(cls_ret);
  }
  return true;
}

ClsRet SoftmaxCls::processSingleItem(const float *logits,
                                     int num_classes) const {
  cv::Mat logit_mat(1, num_classes, CV_32F, const_cast<float *>(logits));

  double max_logit;
  cv::minMaxLoc(logit_mat, nullptr, &max_logit, nullptr, nullptr);

  cv::Mat exp_mat;
  cv::exp(logit_mat - max_logit, exp_mat);

  double sum = cv::sum(exp_mat)[0];

  cv::Mat prob_mat = exp_mat / sum;

  cv::Point class_id_point;
  double score;
  cv::minMaxLoc(prob_mat, nullptr, &score, nullptr, &class_id_point);

  ClsRet cls_ret;
  cls_ret.score = static_cast<float>(score);
  cls_ret.label = class_id_point.x;

  return cls_ret;
}
} // namespace ai_core::dnn
