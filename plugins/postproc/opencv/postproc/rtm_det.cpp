#include "rtm_det.hpp"
#include "logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool RTMDet::processTyped(const TensorData &model_output,
                          const FrameTransformContext &prep_args,
                          const AnchorDetParams &post_args,
                          AlgoOutput &algo_output) const {
  if (model_output.empty()) {
    return false;
  }

  const auto &outputs = model_output;

  const auto &det_output_name = post_args.output_names.at(0);
  const auto &cls_output_name = post_args.output_names.at(1);
  auto det_pred = outputs.at(det_output_name).buffer;
  auto cls_pred = outputs.at(cls_output_name).buffer;

  std::vector<int> det_out_shape = model_output.at(det_output_name).shape;
  std::vector<int> cls_out_shape = model_output.at(cls_output_name).shape;

  int num_classes = cls_out_shape.at(cls_out_shape.size() - 1);
  int anchor_num = det_out_shape.at(det_out_shape.size() - 2);

  DetRet det_ret =
      processSingle(det_pred.getHostPtr<float>(), cls_pred.getHostPtr<float>(),
                    anchor_num, num_classes, prep_args, post_args);

  algo_output.setParams(det_ret);
  return true;
}

bool RTMDet::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const AnchorDetParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {
  if (model_output.empty()) {
    return false;
  }

  const auto &outputs = model_output;

  const auto &det_output_name = post_args.output_names.at(0);
  const auto &cls_output_name = post_args.output_names.at(1);
  auto det_pred = outputs.at(det_output_name).buffer;
  auto cls_pred = outputs.at(cls_output_name).buffer;

  std::vector<int> det_out_shape = model_output.at(det_output_name).shape;
  std::vector<int> cls_out_shape = model_output.at(cls_output_name).shape;

  int batch_size = det_out_shape.at(0);
  int anchor_num = det_out_shape.at(1);
  int num_classes = cls_out_shape.at(2);

  if (prep_args.size() != batch_size) {
    LOG_ERROR_S << "Batch size mismatch between model output (" << batch_size
                << ") and prep_args (" << prep_args.size() << ").";
    return false;
  }

  const float *det_data_ptr = det_pred.getHostPtr<float>();
  const float *cls_data_ptr = cls_pred.getHostPtr<float>();

  algo_output.resize(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    const float *current_det_data = det_data_ptr + i * anchor_num * 4;
    const float *current_cls_data = cls_data_ptr + i * anchor_num * num_classes;
    const auto &current_prep_args = prep_args[i];

    DetRet det_ret =
        processSingle(current_det_data, current_cls_data, anchor_num,
                      num_classes, current_prep_args, post_args);

    algo_output[i].setParams(det_ret);
  }
  return true;
}

DetRet RTMDet::processSingle(const float *det_data_ptr,
                             const float *cls_data_ptr, int anchor_num,
                             int num_classes,
                             const FrameTransformContext &prep_args,
                             const AnchorDetParams &post_args) const {
  std::vector<BBox> results;
  for (int i = 0; i < anchor_num; ++i) {
    auto det_data = det_data_ptr + i * 4;
    auto cls_data = cls_data_ptr + i * num_classes;
    cv::Mat scores(1, num_classes, CV_32F, const_cast<float *>(cls_data));
    cv::Point class_id_point;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id_point);
    if (score > post_args.cond_thre) {
      const Point2f tl = prep_args.mapToSource({det_data[0], det_data[1]});
      const Point2f size = prep_args.mapSizeToSource(det_data[2] - det_data[0],
                                                     det_data[3] - det_data[1]);

      BBox result;
      result.score = score;
      result.label = class_id_point.x;

      result.rect = Rect{static_cast<int>(tl.x), static_cast<int>(tl.y),
                         static_cast<int>(size.x), static_cast<int>(size.y)};
      results.emplace_back(result);
    }
  }

  DetRet det_ret;
  det_ret.bboxes = utils::nms(results, post_args.nms_thre, post_args.cond_thre);
  return det_ret;
}

} // namespace ai_core::dnn
