/**
 * @file yoloDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "nano_det.hpp"
#include "ai_core/logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool NanoDet::process(const TensorData &model_output,
                      const FrameTransformContext &prep_args,
                      const AnchorDetParams &post_args,
                      AlgoOutput &algo_output) const {
  if (model_output.datas.empty()) {
    return false;
  }

  const auto &output_name = post_args.output_names.at(0);
  if (model_output.datas.find(output_name) == model_output.datas.end()) {
    LOG_ERROR_S << "Cannot find output name " << output_name
                << " in model_output.";
    return false;
  }

  auto output = model_output.datas.at(output_name);
  std::vector<int> output_shape = model_output.shapes.at(output_name);

  int num_anchors = output_shape.at(output_shape.size() - 2);
  int stride = output_shape.at(output_shape.size() - 1);
  const float *output_data = output.getHostPtr<float>();

  DetRet det_ret =
      processSingle(output_data, num_anchors, stride, prep_args, post_args);

  algo_output.setParams(det_ret);
  return true;
}

bool NanoDet::batchProcess(const TensorData &model_output,
                           const std::vector<FrameTransformContext> &prep_args,
                           const AnchorDetParams &post_args,
                           std::vector<AlgoOutput> &algo_output) const {
  if (model_output.datas.empty()) {
    return false;
  }

  const auto &output_name = post_args.output_names.at(0);
  if (model_output.datas.find(output_name) == model_output.datas.end()) {
    LOG_ERROR_S << "Cannot find output name " << output_name
                << " in model_output.";
    return false;
  }

  auto output = model_output.datas.at(output_name);
  std::vector<int> output_shape = model_output.shapes.at(output_name);

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

  const auto &input_roi = *prep_args.roi;
  Shape origin_shape;
  if (input_roi.area() > 0) {
    origin_shape.w = input_roi.width;
    origin_shape.h = input_roi.height;
  } else {
    origin_shape = prep_args.origin_shape;
  }
  auto [scaleX, scaleY] = utils::scaleRatio(
      origin_shape, prep_args.model_input_shape, prep_args.is_equal_scale);

  std::vector<BBox> results;
  for (int i = 0; i < raw_data.rows; ++i) {
    const float *data = raw_data.ptr<float>(i);
    // 前 numClasses 个是分数
    cv::Mat scores(1, num_classes, CV_32F, const_cast<float *>(data));
    cv::Point class_id_point;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id_point);

    if (score > post_args.cond_thre) {
      BBox result;
      result.score = score;
      result.label = class_id_point.x;

      // 接下来 4 个是坐标 (x1, y1, x2, y2)
      const float *bbox_data = data + num_classes;
      float x1 = bbox_data[0];
      float y1 = bbox_data[1];
      float x2 = bbox_data[2];
      float y2 = bbox_data[3];

      // 映射原图尺寸
      float w = (x2 - x1) / scaleX;
      float h = (y2 - y1) / scaleY;
      float x = (x1 - prep_args.left_pad) / scaleX + input_roi.x;
      float y = (y1 - prep_args.top_pad) / scaleY + input_roi.y;

      result.rect = cv::Rect(static_cast<int>(x), static_cast<int>(y),
                             static_cast<int>(w), static_cast<int>(h));
      results.push_back(result);
    }
  }

  DetRet det_ret;
  det_ret.bboxes = utils::nms(results, post_args.nms_thre, post_args.cond_thre);
  return det_ret;
}
} // namespace ai_core::dnn
