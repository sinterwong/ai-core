#include "semantic_seg.hpp"
#include "logger.hpp"
#include "ai_core/postprocess_types.hpp"
#include "vision_util.hpp"
#include <opencv2/opencv.hpp>

namespace ai_core::dnn {
bool SemanticSeg::processTyped(const TensorData &model_output,
                               const FrameTransformContext &prep_args,
                               const ConfidenceFilterParams &post_args,
                               AlgoOutput &algo_output) const {
  const auto &feat_map_output_name = post_args.output_names.at(0);
  const auto &feat_map_output = model_output.at(feat_map_output_name).buffer;
  const auto &feat_map_shape = model_output.at(feat_map_output_name).shape;

  const int num_classes = feat_map_shape.at(feat_map_shape.size() - 3);
  const int height = feat_map_shape.at(feat_map_shape.size() - 2);
  const int width = feat_map_shape.at(feat_map_shape.size() - 1);

  if (num_classes > 256) {
    LOG_ERROR_S << "Too many classes for CV_8UC1 mask.";
    return false;
  }

  const float *data = feat_map_output.getHostPtr<float>();
  SegRet seg_ret =
      processSingleItem(data, num_classes, height, width, prep_args, post_args);

  algo_output.setParams(seg_ret);
  return true;
}

bool SemanticSeg::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const ConfidenceFilterParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {
  const auto &feat_map_output_name = post_args.output_names.at(0);
  const auto &feat_map_output = model_output.at(feat_map_output_name).buffer;
  const auto &feat_map_shape = model_output.at(feat_map_output_name).shape;

  if (feat_map_shape.size() != 4) {
    LOG_ERROR_S << "Expected a 4D tensor for batch processing (NCHW), but got "
                << feat_map_shape.size() << " dimensions.";
    return false;
  }

  const int batch_size = feat_map_shape.at(0);
  const int num_classes = feat_map_shape.at(1);
  const int height = feat_map_shape.at(2);
  const int width = feat_map_shape.at(3);

  if (num_classes > 256) {
    LOG_ERROR_S << "Too many classes for CV_8UC1 mask.";
    return false;
  }

  if (batch_size != prep_args.size()) {
    LOG_ERROR_S << "Batch size mismatch between model output (" << batch_size
                << ") and preprocessing arguments (" << prep_args.size()
                << ").";
    return false;
  }

  const float *base_data = feat_map_output.getHostPtr<float>();
  const size_t item_step = static_cast<size_t>(num_classes) * height * width;

  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    const float *current_item_data = base_data + i * item_step;
    const FrameTransformContext &current_item_prep_args = prep_args[i];

    SegRet seg_ret =
        processSingleItem(current_item_data, num_classes, height, width,
                          current_item_prep_args, post_args);

    algo_output[i].setParams(seg_ret);
  }

  return true;
}

SegRet
SemanticSeg::processSingleItem(const float *data, int num_classes, int height,
                               int width,
                               const FrameTransformContext &prep_args,
                               const ConfidenceFilterParams &post_args) const {
  const size_t channel_step = static_cast<size_t>(height) * width;
  cv::Mat class_map(height, width, CV_8UC1);

  if (num_classes == 1) {
    cv::Mat prob_map(height, width, CV_32FC1, const_cast<float *>(data));
    // A single-channel model emits foreground probabilities.
    cv::threshold(prob_map, class_map, post_args.cond_thre, 1,
                  cv::THRESH_BINARY);
    class_map.convertTo(class_map, CV_8U);
  } else {
    // Track the per-pixel argmax across planar class channels.
    cv::Mat max_probs(height, width, CV_32F, const_cast<float *>(data));
    class_map.setTo(0);

    for (int c = 1; c < num_classes; ++c) {
      cv::Mat current_probs(height, width, CV_32F,
                            const_cast<float *>(data + c * channel_step));
      cv::Mat update_mask;
      cv::compare(current_probs, max_probs, update_mask, cv::CMP_GT);

      current_probs.copyTo(max_probs, update_mask);
      class_map.setTo(c, update_mask);
    }

    cv::Mat low_confidence_mask;
    cv::compare(max_probs, post_args.cond_thre, low_confidence_mask,
                cv::CMP_LT);
    class_map.setTo(0, low_confidence_mask);
  }

  SegRet seg_ret;
  seg_ret.cls_to_contours.clear();

  const auto [scaleX, scaleY] = prep_args.scaleRatio();

  if (scaleX <= 0.0f || scaleY <= 0.0f) {
    LOG_ERROR_S << "Invalid scale factors detected: scaleX=" << scaleX
                << ", scaleY=" << scaleY;
    return seg_ret;
  }

  int start_class = 1;
  int end_class = (num_classes == 1) ? 2 : num_classes;

  for (int c = start_class; c < end_class; ++c) {
    cv::Mat class_mask;
    cv::compare(class_map, c, class_mask, cv::CMP_EQ);

    if (cv::countNonZero(class_mask) == 0) {
      continue;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(class_mask, contours, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
      continue;
    }

    for (const auto &contour : contours) {
      Contour transformed_contour;
      transformed_contour.reserve(contour.size());
      std::transform(contour.begin(), contour.end(),
                     std::back_inserter(transformed_contour),
                     [&](const cv::Point &pt) -> Point {
                       const Point2f src =
                           prep_args.mapToSource({static_cast<float>(pt.x),
                                                  static_cast<float>(pt.y)});
                       return Point{cvRound(src.x), cvRound(src.y)};
                     });

      seg_ret.cls_to_contours[c].emplace_back(std::move(transformed_contour));
    }
  }

  return seg_ret;
}
} // namespace ai_core::dnn
