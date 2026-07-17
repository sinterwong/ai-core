/**
 * @file ncnn_generic_preprocessor.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ncnn_generic_preprocessor.hpp"

#include "ai_core/opencv_interop.hpp"

#include "ai_core/logger.hpp"
#include <algorithm>
#include <ncnn/mat.h>
#include <ncnn/option.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace ai_core::dnn::mncnn {

TypedBuffer
NcnnGenericPreprocessor::process(const FramePreprocessArg &args,
                                 const FrameInput &input,
                                 FrameTransformContext &runtime_args) const {

  if (args.output_location != BufferLocation::CPU) {
    LOG_WARNING_S
        << "NCNN NcnnGenericPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }
  if (input.image.empty()) {
    LOG_ERROR_S << "Input frame is empty.";
    throw std::runtime_error("Input frame is empty.");
  }

  runtime_args.roi =
      input.roi.value_or(Rect{0, 0, input.image.width, input.image.height});
  runtime_args.origin_shape = {input.image.width, input.image.height,
                               input.image.channels()};

  const cv::Mat cv_image_orig = interop::matFromView(input.image);
  const cv::Rect input_roi = interop::toCv(runtime_args.roi);
  if (input_roi.x < 0 || input_roi.y < 0 || input_roi.width <= 0 ||
      input_roi.height <= 0 ||
      input_roi.x + input_roi.width > cv_image_orig.cols ||
      input_roi.y + input_roi.height > cv_image_orig.rows) {
    LOG_ERROR_S << "Invalid ROI: " << input_roi
                << " for image size: " << cv_image_orig.size();
    throw std::runtime_error("Invalid ROI.");
  }

  if (cv_image_orig.empty()) {
    LOG_ERROR_S << "Input cv::Mat image is empty.";
    throw std::runtime_error("Input cv::Mat image is empty.");
  }

  int target_width = args.model_input_shape.w;
  int target_height = args.model_input_shape.h;
  int target_channels = cv_image_orig.channels();

  if (target_width <= 0 || target_height <= 0 || target_channels <= 0) {
    LOG_ERROR_S << "Invalid target dimensions in FramePreprocessArg: W="
                << target_width << ", H=" << target_height
                << ", C=" << target_channels;
    throw std::runtime_error(
        "Invalid target dimensions in FramePreprocessArg.");
  }

  cv::Mat current_cv_mat = cv_image_orig;

  if (input_roi.area() > 0) {
    cv::Rect valid_roi =
        input_roi & cv::Rect(0, 0, current_cv_mat.cols, current_cv_mat.rows);
    if (valid_roi.area() > 0) {
      current_cv_mat = current_cv_mat(valid_roi).clone();
    } else {
      LOG_WARNING_S
          << "Specified ROI is outside image bounds or has zero area. "
             "Using full image.";
    }
  }

  ncnn::Mat ncnn_in;
  int ncnn_pixel_type = ncnn::Mat::PIXEL_BGR;
  if (target_channels == 3) {
    if (current_cv_mat.channels() == 3) {
      ncnn_pixel_type = ncnn::Mat::PIXEL_BGR2RGB;
    } else if (current_cv_mat.channels() == 1) {
      ncnn_pixel_type = ncnn::Mat::PIXEL_GRAY2RGB;
    } else {
      LOG_ERROR_S << "Unsupported channel count " << current_cv_mat.channels()
                  << " for 3-channel RGB output.";
      throw std::runtime_error("Unsupported channel count for RGB output.");
    }
  } else if (target_channels == 1) {
    if (current_cv_mat.channels() == 3)
      ncnn_pixel_type = ::ncnn::Mat::PIXEL_BGR2GRAY;
    else if (current_cv_mat.channels() == 1)
      ncnn_pixel_type = ncnn::Mat::PIXEL_GRAY;
    else {
      LOG_ERROR_S << "Unsupported channel count " << current_cv_mat.channels()
                  << " for 1-channel Gray output.";
      throw std::runtime_error("Unsupported channel count for Gray output.");
    }
  } else {
    LOG_ERROR_S << "Unsupported target channel count: " << target_channels;
    throw std::runtime_error("Unsupported target channel count.");
  }

  // Resize & Pad
  if (args.need_resize) {
    if (args.is_equal_scale) {
      int img_width = current_cv_mat.cols;
      int img_height = current_cv_mat.rows;
      float scale = std::min((float)target_width / img_width,
                             (float)target_height / img_height);
      int scaled_width = static_cast<int>(img_width * scale);
      int scaled_height = static_cast<int>(img_height * scale);

      ncnn::Mat temp_ncnn_mat = ncnn::Mat::from_pixels_resize(
          current_cv_mat.data, ncnn_pixel_type, current_cv_mat.cols,
          current_cv_mat.rows, scaled_width, scaled_height);

      runtime_args.left_pad = (target_width - scaled_width) / 2;
      runtime_args.top_pad = (target_height - scaled_height) / 2;
      int right_pad = target_width - scaled_width - runtime_args.left_pad;
      int bottom_pad = target_height - scaled_height - runtime_args.top_pad;

      ncnn::copy_make_border(temp_ncnn_mat, ncnn_in, runtime_args.top_pad,
                             bottom_pad, runtime_args.left_pad, right_pad,
                             ncnn::BORDER_CONSTANT, (float)args.pad[0]);
      if (ncnn_in.w != target_width || ncnn_in.h != target_height) {
        LOG_WARNING_S << "Padded NCNN Mat size (" << ncnn_in.w << "x"
                      << ncnn_in.h << ") mismatch target (" << target_width
                      << "x" << target_height
                      << "). This is unexpected after ncnn::copy_make_border.";
      }

    } else {
      ncnn_in = ncnn::Mat::from_pixels_resize(
          current_cv_mat.data, ncnn_pixel_type, current_cv_mat.cols,
          current_cv_mat.rows, target_width, target_height);
    }
  } else {
    if (current_cv_mat.cols != target_width ||
        current_cv_mat.rows != target_height) {
      LOG_WARNING_S << "NeedResize is false, but image dimensions ("
                    << current_cv_mat.cols << "x" << current_cv_mat.rows
                    << ") do not match target (" << target_width << "x"
                    << target_height
                    << "). Resizing to target dimensions as a fallback.";
      ncnn_in = ncnn::Mat::from_pixels_resize(
          current_cv_mat.data, ncnn_pixel_type, current_cv_mat.cols,
          current_cv_mat.rows, target_width, target_height);
    } else {
      ncnn_in =
          ncnn::Mat::from_pixels(current_cv_mat.data, ncnn_pixel_type,
                                 current_cv_mat.cols, current_cv_mat.rows);
    }
  }

  if (ncnn_in.empty()) {
    LOG_ERROR_S << "ncnn::Mat is empty after conversion/resize.";
    throw std::runtime_error("ncnn::Mat is empty after conversion/resize.");
  }

  // Normalization (mean subtraction and scaling)
  if (!args.mean_vals.empty() || !args.norm_vals.empty()) {
    if (args.mean_vals.size() != ncnn_in.c && !args.mean_vals.empty()) {
      LOG_ERROR_S << "mean_vals size (" << args.mean_vals.size()
                  << ") != ncnnIn.c (" << ncnn_in.c << ")";
      throw std::runtime_error(
          "MeanVals size mismatch with NCNN Mat channels.");
    }
    if (args.norm_vals.size() != ncnn_in.c && !args.norm_vals.empty()) {
      LOG_ERROR_S << "norm_vals size (" << args.norm_vals.size()
                  << ") != ncnnIn.c (" << ncnn_in.c << ")";
      throw std::runtime_error(
          "NormVals size mismatch with NCNN Mat channels.");
    }

    std::vector<float> ncnn_norm_vals = args.norm_vals;
    if (!ncnn_norm_vals.empty()) {
      std::transform(ncnn_norm_vals.begin(), ncnn_norm_vals.end(),
                     ncnn_norm_vals.begin(),
                     [](float val) { return val == 0.0f ? 1.0f : 1.0f / val; });
    } else {
      if (!args.mean_vals.empty()) {
        ncnn_norm_vals.assign(ncnn_in.c, 1.0f);
      }
    }

    std::vector<float> ncnn_mean_vals = args.mean_vals;
    if (ncnn_mean_vals.empty() && !ncnn_norm_vals.empty()) {
      ncnn_mean_vals.assign(ncnn_in.c, 0.0f);
    }

    ncnn_in.substract_mean_normalize(
        ncnn_mean_vals.empty() ? nullptr : ncnn_mean_vals.data(),
        ncnn_norm_vals.empty() ? nullptr : ncnn_norm_vals.data());
  }

  if (ncnn_in.elemsize != sizeof(float)) {
    LOG_ERROR_S << "NCNN Mat elemsize is not float. Unexpected data format.";
    throw std::runtime_error("NCNN Mat elemsize is not float.");
  }

  const size_t byte_size = ncnn_in.total() * sizeof(float);
  const uint8_t *ncnn_data_ptr =
      reinterpret_cast<const uint8_t *>(ncnn_in.data);

  std::vector<uint8_t> cpu_data_vec(ncnn_data_ptr, ncnn_data_ptr + byte_size);

  TypedBuffer output_buffer =
      TypedBuffer::createFromCpu(DataType::FLOAT32, std::move(cpu_data_vec));
  return output_buffer;
}

TypedBuffer NcnnGenericPreprocessor::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &input,
    std::vector<FrameTransformContext> &runtime_context) const {
  LOG_ERROR_S << "NCNN batch preprocessor not implemented.";
  throw std::runtime_error("NCNN batch preprocessor not implemented.");
}

}; // namespace ai_core::dnn::mncnn
