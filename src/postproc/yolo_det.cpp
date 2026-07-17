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
#include "yolo_det.hpp"
#include "ai_core/logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool Yolov11Det::processTyped(const TensorData &model_output,
                              const FrameTransformContext &prep_args,
                              const AnchorDetParams &post_args,
                              AlgoOutput &algo_output) const {
  const auto &output_shapes = model_output.shapes;
  const auto &input_shape = prep_args.model_input_shape;
  const auto &outputs = model_output.datas;

  // just one output
  if (outputs.size() != 1) {
    LOG_ERROR_S << "AnchorDetParams(Yolov11Det) unexpected size of outputs "
                << outputs.size();
    throw std::runtime_error(
        "AnchorDetParams(Yolov11Det)  unexpected size of outputs");
  }
  auto output = outputs.at(post_args.output_names.at(0));

  std::vector<int> output_shape =
      output_shapes.at(post_args.output_names.at(0));
  int signal_result_num = output_shape.at(output_shape.size() - 2);
  int stride_num = output_shape.at(output_shape.size() - 1);

  cv::Mat raw_data = cv::Mat(stride_num, signal_result_num, CV_32F);
  if (output.dataType() == DataType::FLOAT32) {
    cv::transpose(cv::Mat(signal_result_num, stride_num, CV_32F,
                          const_cast<void *>(output.getRawHostPtr())),
                  raw_data);
  } else if (output.dataType() == DataType::FLOAT16) {
    const uint16_t *fp16_data = output.getHostPtr<uint16_t>();
    cv::Mat half_mat(1, output.getElementCount(), CV_16F, (void *)fp16_data);
    cv::Mat float_mat(1, output.getElementCount(), CV_32F);
    half_mat.convertTo(float_mat, CV_32F);
    cv::transpose(
        cv::Mat(signal_result_num, stride_num, CV_32F, float_mat.data),
        raw_data);
  }

  std::vector<BBox> results = processRawOutput(
      raw_data, input_shape, prep_args, post_args, signal_result_num - 4);

  DetRet det_ret;
  det_ret.bboxes = utils::nms(results, post_args.nms_thre, post_args.cond_thre);
  algo_output.setParams(det_ret);
  return true;
}

bool Yolov11Det::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const AnchorDetParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {
  const auto &outputs = model_output.datas;
  if (outputs.size() != 1) {
    LOG_ERROR_S << "Yolov11Det::batchProcess unexpected size of outputs: "
                << outputs.size();
    throw std::runtime_error(
        "Yolov11Det::batchProcess expects only 1 output tensor.");
  }
  const auto &output_tensor = outputs.at(post_args.output_names.at(0));
  const auto &output_shape =
      model_output.shapes.at(post_args.output_names.at(0));

  if (output_shape.size() != 3) {
    LOG_ERROR_S << "Yolov11Det::batchProcess unexpected output dimensions: "
                << output_shape.size();
    throw std::runtime_error("Yolov11Det::batchProcess expects a 3D output "
                             "tensor [batch, stride, num_results].");
  }
  const int batch_size = output_shape.at(0);
  const int stride_num = output_shape.at(1);
  const int signal_result_num = output_shape.at(2);
  const int num_classes = stride_num - 4;

  if (batch_size != prep_args.size()) {
    LOG_ERROR_S
        << "Yolov11Det::batchProcess mismatch between model output batch size ("
        << batch_size << ") and prep_args size (" << prep_args.size() << ").";
    throw std::runtime_error(
        "Batch size mismatch in Yolov11Det::batchProcess.");
  }

  algo_output.resize(batch_size);
  const size_t elements_per_sample =
      static_cast<size_t>(stride_num) * signal_result_num;

  const float *batched_float_data = nullptr;
  cv::Mat full_float_mat;
  if (output_tensor.dataType() == DataType::FLOAT32) {
    batched_float_data = output_tensor.getHostPtr<float>();
  } else if (output_tensor.dataType() == DataType::FLOAT16) {
    const uint16_t *fp16_data = output_tensor.getHostPtr<uint16_t>();
    cv::Mat half_mat(1, output_tensor.getElementCount(), CV_16F,
                     const_cast<uint16_t *>(fp16_data));
    half_mat.convertTo(full_float_mat, CV_32F);
    batched_float_data = full_float_mat.ptr<float>();
  } else {
    throw std::runtime_error(
        "Unsupported data type in Yolov11Det::batchProcess.");
  }

  for (int i = 0; i < batch_size; ++i) {
    const float *current_sample_data =
        batched_float_data + i * elements_per_sample;

    cv::Mat single_output_mat(stride_num, signal_result_num, CV_32F,
                              const_cast<float *>(current_sample_data));
    cv::Mat transposed_output;
    cv::transpose(single_output_mat, transposed_output);

    const auto &current_prep_args = prep_args[i];
    const auto &input_shape = current_prep_args.model_input_shape;

    std::vector<BBox> results =
        processRawOutput(transposed_output, input_shape, current_prep_args,
                         post_args, num_classes);

    DetRet det_ret;
    det_ret.bboxes =
        utils::nms(results, post_args.nms_thre, post_args.cond_thre);
    algo_output[i].setParams(det_ret);
  }
  return true;
}

std::vector<BBox> Yolov11Det::processRawOutput(
    const cv::Mat &transposed_output, const Shape &input_shape,
    const FrameTransformContext &prep_args, const AnchorDetParams &post_args,
    int num_classes) const {
  std::vector<BBox> results;
  Shape origin_shape;
  const auto &input_roi = prep_args.roi;
  if (input_roi.area() > 0) {
    origin_shape.w = input_roi.width;
    origin_shape.h = input_roi.height;
  } else {
    origin_shape = prep_args.origin_shape;
  }
  auto [scaleX, scaleY] =
      utils::scaleRatio(origin_shape, input_shape, prep_args.is_equal_scale);

  for (int i = 0; i < transposed_output.rows; ++i) {
    const float *data = transposed_output.ptr<float>(i);

    cv::Mat scores(1, num_classes, CV_32F, (void *)(data + 4));
    cv::Point class_id_point;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id_point);

    if (score > post_args.cond_thre) {
      BBox result;
      result.score = score;
      result.label = class_id_point.x;

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      x = x - 0.5 * w;
      y = y - 0.5 * h;

      if (prep_args.is_equal_scale) {
        x = (x - prep_args.left_pad) / scaleX;
        y = (y - prep_args.top_pad) / scaleY;
      } else {
        x = x / scaleX;
        y = y / scaleY;
      }
      w = w / scaleX;
      h = h / scaleY;
      x += input_roi.x;
      y += input_roi.y;
      result.rect = Rect{static_cast<int>(x), static_cast<int>(y),
                         static_cast<int>(w), static_cast<int>(h)};

      results.push_back(result);
    }
  }

  return results;
}
} // namespace ai_core::dnn
