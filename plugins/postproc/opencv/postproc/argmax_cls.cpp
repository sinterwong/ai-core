#include "argmax_cls.hpp"
#include "ai_core/logger.hpp"
#include <opencv2/core.hpp>
#include <utility>

namespace ai_core::dnn {
bool ArgmaxCls::processTyped(const TensorData &model_output,
                             const FrameTransformContext &prep_args,
                             const GenericPostParams &post_args,
                             AlgoOutput &algo_output) const {
  if (model_output.empty()) {
    LOG_ERROR_S << "model_output is empty";
    return false;
  }

  const auto &score_output_name = post_args.output_names.at(0);
  const auto &output = model_output.at(score_output_name).buffer;
  const auto &output_shape = model_output.at(score_output_name).shape;

  const int num_classes = output_shape.at(output_shape.size() - 1);

  ClsRet cls_ret = processSingleItem(output.getHostPtr<float>(), num_classes,
                                     post_args.keep_class_probs);

  algo_output.setParams(std::move(cls_ret));
  return true;
}

bool ArgmaxCls::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const GenericPostParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {
  if (model_output.empty()) {
    LOG_ERROR_S << "model_output is empty";
    return false;
  }

  const auto &score_output_name = post_args.output_names.at(0);
  const auto &output = model_output.at(score_output_name).buffer;
  const auto &output_shape = model_output.at(score_output_name).shape;

  if (output_shape.size() != 2) {
    LOG_ERROR_S
        << "Expected a 2D tensor for batch classification (N, C), but got "
        << output_shape.size() << " dimensions.";
    return false;
  }

  const int batch_size = output_shape.at(0);
  const int num_classes = output_shape.at(1);

  const float *base_scores = output.getHostPtr<float>();

  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    ClsRet cls_ret = processSingleItem(base_scores + i * num_classes,
                                       num_classes, post_args.keep_class_probs);
    algo_output[i].setParams(std::move(cls_ret));
  }
  return true;
}

ClsRet ArgmaxCls::processSingleItem(const float *scores, int num_classes,
                                    bool keep_probs) const {
  const cv::Mat score_mat(1, num_classes, CV_32F, const_cast<float *>(scores));

  cv::Point class_id_point;
  double score;
  cv::minMaxLoc(score_mat, nullptr, &score, nullptr, &class_id_point);

  ClsRet cls_ret;
  // The score is passed through as-is: the model already normalized it.
  cls_ret.score = static_cast<float>(score);
  cls_ret.label = class_id_point.x;
  if (keep_probs) {
    cls_ret.probs.assign(scores, scores + num_classes);
  }

  return cls_ret;
}
} // namespace ai_core::dnn
