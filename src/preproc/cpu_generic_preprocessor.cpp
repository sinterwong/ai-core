/**
 * @file cpu_generic_preprocessor.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "cpu_generic_preprocessor.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

namespace ai_core::dnn::cpu {

TypedBuffer
CpuGenericCvPreprocessor::process(const FramePreprocessArg &params,
                                  const FrameInput &frame_input,
                                  FrameTransformContext &runtime_args) const {

  if (params.output_location != BufferLocation::CPU) {
    LOG_WARNING_S
        << "CPU CpuGenericCvPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }

  cv::Mat normalized_image =
      preprocessSingleFrame(params, frame_input, runtime_args);

  const int input_channels = normalized_image.channels();

  switch (params.data_type) {
  case DataType::FLOAT32:
    return preprocessFP32(normalized_image, input_channels,
                          params.model_input_shape.h,
                          params.model_input_shape.w, params.hwc2chw);
  case DataType::FLOAT16:
    return preprocessFP16(normalized_image, input_channels,
                          params.model_input_shape.h,
                          params.model_input_shape.w, params.hwc2chw);
  default:
    LOG_ERROR_S << "Unsupported data type: "
                << static_cast<int>(params.data_type);
    throw std::runtime_error("Unsupported data type");
  }
}

TypedBuffer CpuGenericCvPreprocessor::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtime_args) const {
  if (args.output_location != BufferLocation::CPU) {
    LOG_WARNING_S
        << "CPU CpuGenericCvPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }

  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batch_size = frames.size();
  runtime_args.resize(batch_size);

  std::vector<cv::Mat> processed_images;
  processed_images.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    runtime_args[i].model_input_shape = args.model_input_shape;
    runtime_args[i].is_equal_scale = args.is_equal_scale;
    processed_images.push_back(
        preprocessSingleFrame(args, frames[i], runtime_args[i]));
  }

  const int input_channels = args.model_input_shape.c;
  const int input_height = args.model_input_shape.h;
  const int input_width = args.model_input_shape.w;
  const size_t single_image_size =
      static_cast<size_t>(input_channels) * input_height * input_width;
  const size_t total_elements = single_image_size * batch_size;

  // 合并所有处理后的图像到一个缓冲区中
  switch (args.data_type) {
  case DataType::FLOAT32: {
    TypedBuffer result;
    result.resize(total_elements);
    float *dst_ptr = result.getHostPtr<float>();

    for (const auto &image : processed_images) {
      convertLayout(image, dst_ptr, args.hwc2chw);
      dst_ptr += single_image_size; // 移动指针到下一个图像的位置
    }
    return result;
  }
  case DataType::FLOAT16: {
    std::vector<float> batch_data_f_p32(total_elements);
    float *dst_ptr = batch_data_f_p32.data();

    for (const auto &image : processed_images) {
      convertLayout(image, dst_ptr, args.hwc2chw);
      dst_ptr += single_image_size;
    }

    const float fp16_max_value = 65504.0f;
    for (float &val : batch_data_f_p32) {
      val = std::clamp(val, -fp16_max_value, fp16_max_value);
    }

    cv::Mat float_mat(1, static_cast<int>(total_elements), CV_32F,
                      batch_data_f_p32.data());
    cv::Mat half_mat;
    float_mat.convertTo(half_mat, CV_16F);

    const size_t byte_size = total_elements * sizeof(uint16_t);
    const uint8_t *start_ptr = half_mat.data;
    std::vector<uint8_t> final_data(start_ptr, start_ptr + byte_size);

    return TypedBuffer::createFromCpu(DataType::FLOAT16, std::move(final_data));
  }
  default:
    LOG_ERROR_S << "Unsupported data type: "
                << static_cast<int>(args.data_type);
    throw std::runtime_error("Unsupported data type");
  }
}

cv::Mat CpuGenericCvPreprocessor::preprocessSingleFrame(
    const FramePreprocessArg &params, const FrameInput &frame_input,
    FrameTransformContext &runtime_args) const {
  if (frame_input.image.empty()) {
    LOG_ERROR_S << "Input frame is empty.";
    throw std::runtime_error("Input frame is empty.");
  }

  const cv::Mat image = interop::matFromView(frame_input.image);

  runtime_args.roi = frame_input.roi.value_or(
      Rect{0, 0, frame_input.image.width, frame_input.image.height});
  runtime_args.origin_shape = {frame_input.image.width,
                               frame_input.image.height,
                               frame_input.image.channels()};

  const cv::Rect roi = interop::toCv(runtime_args.roi);
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    LOG_ERROR_S << "Invalid ROI: " << roi
                << " for image size: " << image.size();
    throw std::runtime_error("Invalid ROI.");
  }

  cv::Mat cropped_image = image(roi).clone();

  cv::Mat resized_image;
  if (params.need_resize) {
    if (params.is_equal_scale) {
      if (params.pad.size() > 4) {
        LOG_WARNING_S << "Padding vector has more than 4 elements. Only the "
                         "first 4 will be used.";
      }
      cv::Scalar pad = utils::createScalarFromVector(params.pad);
      auto pad_ret = utils::escaleResizeWithPad(
          cropped_image, resized_image, params.model_input_shape.h,
          params.model_input_shape.w, pad);
      runtime_args.top_pad = pad_ret.h;
      runtime_args.left_pad = pad_ret.w;
    } else {
      cv::resize(
          cropped_image, resized_image,
          cv::Size(params.model_input_shape.w, params.model_input_shape.h), 0,
          0, cv::INTER_LINEAR);
    }
  } else {
    resized_image = cropped_image;
  }

  cv::Mat float_image;
  resized_image.convertTo(float_image, CV_32F);

  cv::Mat normalized_image = float_image;
  if (!params.mean_vals.empty() && !params.norm_vals.empty()) {
    const int input_channels = normalized_image.channels();
    if (params.mean_vals.size() != input_channels ||
        params.norm_vals.size() != input_channels) {
      throw std::runtime_error(
          "mean_vals and norm_vals size must match input channels");
    }

    std::vector<cv::Mat> channels(input_channels);
    cv::split(normalized_image, channels);

    for (int i = 0; i < input_channels; ++i) {
      channels[i] = (channels[i] - params.mean_vals[i]) / params.norm_vals[i];
    }
    cv::merge(channels, normalized_image);
  }

  return normalized_image;
}

void CpuGenericCvPreprocessor::convertLayout(const cv::Mat &image, float *dst,
                                             bool hwc2chw) const {
  const int height = image.rows;
  const int width = image.cols;
  const int channels = image.channels();
  const size_t total_elements = static_cast<size_t>(height) * width * channels;

  if (!hwc2chw || channels == 1) {
    if (image.isContinuous()) {
      std::memcpy(dst, image.data, total_elements * sizeof(float));
    } else {
      const size_t row_size = width * channels * sizeof(float);
      for (int h = 0; h < height; ++h) {
        std::memcpy(dst + h * width * channels, image.ptr<float>(h), row_size);
      }
    }
  } else {
    const int plane_size = height * width;
    for (int h = 0; h < height; ++h) {
      const float *row_ptr = image.ptr<float>(h);
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          dst[c * plane_size + h * width + w] = row_ptr[w * channels + c];
        }
      }
    }
  }
}

TypedBuffer
CpuGenericCvPreprocessor::preprocessFP32(const cv::Mat &normalized_image,
                                         int input_channels, int input_height,
                                         int input_width, bool hwc2chw) const {
  TypedBuffer result;

  const size_t total_elements =
      static_cast<size_t>(input_channels) * input_height * input_width;
  result.resize(total_elements);

  float *data_ptr = result.getHostPtr<float>();

  convertLayout(normalized_image, data_ptr, hwc2chw);

  return result;
}

TypedBuffer
CpuGenericCvPreprocessor::preprocessFP16(const cv::Mat &normalized_image,
                                         int input_channels, int input_height,
                                         int input_width, bool hwc2chw) const {
  const size_t total_elements =
      static_cast<size_t>(input_channels) * input_height * input_width;

  std::vector<float> tensor_data_f_p32(total_elements);
  convertLayout(normalized_image, tensor_data_f_p32.data(), hwc2chw);

  const float fp16_max_value = 65504.0f;
  for (float &val : tensor_data_f_p32) {
    val = std::clamp(val, -fp16_max_value, fp16_max_value);
  }

  cv::Mat float_mat(1, static_cast<int>(total_elements), CV_32F,
                    tensor_data_f_p32.data());
  cv::Mat half_mat;
  float_mat.convertTo(half_mat, CV_16F);

  const size_t byte_size = total_elements * sizeof(uint16_t);
  const uint8_t *start_ptr = half_mat.data;
  std::vector<uint8_t> final_data(start_ptr, start_ptr + byte_size);

  return TypedBuffer::createFromCpu(DataType::FLOAT16, std::move(final_data));
}
} // namespace ai_core::dnn::cpu
