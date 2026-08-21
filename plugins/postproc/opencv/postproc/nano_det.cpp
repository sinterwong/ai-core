#include "nano_det.hpp"
#include "ai_core/logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool NanoDet::processTyped(const TensorData &model_output,
                           const FrameTransformContext &prep_args,
                           const AnchorDetParams &post_args,
                           AlgoOutput &algo_output) const {
  if (model_output.empty()) {
    return false;
  }

  const auto &output_name = post_args.output_names.at(0);
  if (!model_output.contains(output_name)) {
    LOG_ERROR_S << "Cannot find output name " << output_name
                << " in model_output.";
    return false;
  }

  auto output = model_output.at(output_name).buffer;
  std::vector<int> output_shape = model_output.at(output_name).shape;

  int num_anchors = output_shape.at(output_shape.size() - 2);
  int stride = output_shape.at(output_shape.size() - 1);
  const float *output_data = output.getHostPtr<float>();

  DetRet det_ret =
      processSingle(output_data, num_anchors, stride, prep_args, post_args);

  algo_output.setParams(det_ret);
  return true;
}

bool NanoDet::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const AnchorDetParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {
  if (model_output.empty()) {
    return false;
  }

  const auto &output_name = post_args.output_names.at(0);
  if (!model_output.contains(output_name)) {
    LOG_ERROR_S << "Cannot find output name " << output_name
                << " in model_output.";
    return false;
  }

  auto output = model_output.at(output_name).buffer;
  std::vector<int> output_shape = model_output.at(output_name).shape;

  if (output_shape.size() != 3) {
    LOG_ERROR_S
        << "Batch process expects output tensor with 3 dimensions, but got "
        << output_shape.size();
    return false;
  }

  int batch_size = output_shape.at(0);
  int num_anchors = output_shape.at(1);
  int stride = output_shape.at(2);

  if (prep_args.size() != batch_size) {
    LOG_ERROR_S << "Batch size mismatch between model output (" << batch_size
                << ") and prep_args (" << prep_args.size() << ").";
    return false;
  }

  const float *all_output_data = output.getHostPtr<float>();
  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    const float *current_item_data = all_output_data + i * num_anchors * stride;
    const auto &current_item_prep_args = prep_args.at(i);

    DetRet det_ret = processSingle(current_item_data, num_anchors, stride,
                                   current_item_prep_args, post_args);

    algo_output[i].setParams(det_ret);
  }

  return true;
}

DetRet NanoDet::processSingle(const float *output_data, int num_anchors,
                              int stride,
                              const FrameTransformContext &prep_args,
                              const AnchorDetParams &post_args) const {
  cv::Mat raw_data(num_anchors, stride, CV_32F,
                   const_cast<float *>(output_data));
  int num_classes = stride - 4;

  std::vector<BBox> results;
  for (int i = 0; i < raw_data.rows; ++i) {
    const float *data = raw_data.ptr<float>(i);
    // NanoDet packs class scores before the four box coordinates.
    cv::Mat scores(1, num_classes, CV_32F, const_cast<float *>(data));
    cv::Point class_id_point;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id_point);

    if (score > post_args.cond_thre) {
      BBox result;
      result.score = score;
      result.label = class_id_point.x;

      const float *bbox_data = data + num_classes;

      const Point2f tl = prep_args.mapToSource({bbox_data[0], bbox_data[1]});
      const Point2f size = prep_args.mapSizeToSource(
          bbox_data[2] - bbox_data[0], bbox_data[3] - bbox_data[1]);

      result.rect = Rect{static_cast<int>(tl.x), static_cast<int>(tl.y),
                         static_cast<int>(size.x), static_cast<int>(size.y)};
      results.push_back(result);
    }
  }

  DetRet det_ret;
  det_ret.bboxes = utils::nms(results, post_args.nms_thre, post_args.cond_thre);
  return det_ret;
}
} // namespace ai_core::dnn
