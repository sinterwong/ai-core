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
bool FprCls::process(const TensorData &model_output,
                     const FrameTransformContext &prep_args,
                     const GenericPostParams &post_args,
                     AlgoOutput &algo_output) const {
  const auto &score_output_name = post_args.output_names.at(0);
  const auto &birad_output_name = post_args.output_names.at(1);

  const auto &outputs = model_output.datas;
  auto p_scores = outputs.at(score_output_name);
  auto p_birads = outputs.at(birad_output_name);

  std::vector<int> p_scores_shape = model_output.shapes.at(score_output_name);
  int num_classes = p_scores_shape.at(p_scores_shape.size() - 1);

  std::vector<int> p_birads_shape = model_output.shapes.at(birad_output_name);
  int num_birads = p_birads_shape.at(p_birads_shape.size() - 1);

  FprClsRet fpr_ret =
      processSingleItem(p_scores.getHostPtr<float>(), num_classes,
                        p_birads.getHostPtr<float>(), num_birads);

  algo_output.setParams(fpr_ret);
  return true;
}

bool FprCls::batchProcess(const TensorData &model_output,
                          const std::vector<FrameTransformContext> &prep_args,
                          const GenericPostParams &post_args,
                          std::vector<AlgoOutput> &algo_output) const {
  const auto &score_output_name = post_args.output_names.at(0);
  const auto &birad_output_name = post_args.output_names.at(1);

  const auto &outputs = model_output.datas;
  auto p_scores = outputs.at(score_output_name);
  auto p_birads = outputs.at(birad_output_name);

  std::vector<int> p_scores_shape = model_output.shapes.at(score_output_name);
  int batch_size = p_scores_shape.at(0);
  int num_classes = p_scores_shape.at(p_scores_shape.size() - 1);

  std::vector<int> p_birads_shape = model_output.shapes.at(birad_output_name);
  int num_birads = p_birads_shape.at(p_birads_shape.size() - 1);

  const float *scores_data = p_scores.getHostPtr<float>();
  const float *birads_data = p_birads.getHostPtr<float>();

  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    const float *current_scores = scores_data + i * num_classes;
    const float *current_birads = birads_data + i * num_birads;

    FprClsRet fpr_ret = processSingleItem(current_scores, num_classes,
                                          current_birads, num_birads);
    algo_output[i].setParams(fpr_ret);
  }
  return true;
}

FprClsRet FprCls::processSingleItem(const float *scores_data, int num_classes,
                                    const float *birads_data,
                                    int num_birads) const {
  cv::Mat scores(1, num_classes, CV_32F, const_cast<float *>(scores_data));
  cv::Mat birads(1, num_birads, CV_32F, const_cast<float *>(birads_data));

  cv::Point class_id_point;
  double score;
  cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id_point);

  cv::Point birads_id_point;
  double birads_score;
  cv::minMaxLoc(birads, nullptr, &birads_score, nullptr, &birads_id_point);

  FprClsRet fpr_ret;
  fpr_ret.score = score;
  fpr_ret.label = class_id_point.x;
  fpr_ret.score_probs.assign(scores_data, scores_data + num_classes);
  fpr_ret.birad = birads_id_point.x;

  return fpr_ret;
}
} // namespace ai_core::dnn
