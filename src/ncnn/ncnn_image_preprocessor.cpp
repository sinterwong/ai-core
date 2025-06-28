/**
 * @file ncnn_image_preprocessor.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ncnn_image_preprocessor.hpp"
#include "logger.hpp"
#include "vision_util.hpp"
#include <algorithm>
#include <ncnn/mat.h>
#include <ncnn/option.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace ai_core::dnn::mncnn {

TypedBuffer ImagePreprocessor::process(FramePreprocessArg &args,
                                       const FrameInput &frameData) const {
  const cv::Mat &cv_image_orig = frameData.image;
  if (cv_image_orig.empty()) {
    LOG_ERRORS << "Input cv::Mat image is empty.";
    throw std::runtime_error("Input cv::Mat image is empty.");
  }

  int target_w = args.modelInputShape.w;
  int target_h = args.modelInputShape.h;
  int target_c = frameData.image.channels();

  if (target_w <= 0 || target_h <= 0 || target_c <= 0) {
    LOG_ERRORS << "Invalid target dimensions in FramePreprocessArg: W="
               << target_w << ", H=" << target_h << ", C=" << target_c;
    throw std::runtime_error(
        "Invalid target dimensions in FramePreprocessArg.");
  }

  cv::Mat current_cv_mat = cv_image_orig;

  if (args.roi.area() > 0) {
    cv::Rect valid_roi =
        args.roi & cv::Rect(0, 0, current_cv_mat.cols, current_cv_mat.rows);
    if (valid_roi.area() > 0) {
      current_cv_mat = current_cv_mat(valid_roi).clone();
    } else {
      LOG_WARNINGS << "Specified ROI is outside image bounds or has zero area. "
                      "Using full image.";
    }
  }

  ncnn::Mat ncnn_in;
  int ncnn_pixel_type = ncnn::Mat::PIXEL_BGR;
  if (target_c == 3) {
    if (current_cv_mat.channels() == 3)
      ncnn_pixel_type =
          ncnn::Mat::PIXEL_BGR2RGB; // Convert BGR cv::Mat to RGB ncnn::Mat
    else if (current_cv_mat.channels() == 1)
      ncnn_pixel_type =
          ncnn::Mat::PIXEL_GRAY2RGB; // Convert Gray cv::Mat to RGB ncnn::Mat
    else {
      LOG_ERRORS << "Unsupported channel count " << current_cv_mat.channels()
                 << " for 3-channel RGB output.";
      throw std::runtime_error("Unsupported channel count for RGB output.");
    }
  } else if (target_c == 1) {
    if (current_cv_mat.channels() == 3)
      ncnn_pixel_type = ::ncnn::Mat::PIXEL_BGR2GRAY;
    else if (current_cv_mat.channels() == 1)
      ncnn_pixel_type = ncnn::Mat::PIXEL_GRAY;
    else {
      LOG_ERRORS << "Unsupported channel count " << current_cv_mat.channels()
                 << " for 1-channel Gray output.";
      throw std::runtime_error("Unsupported channel count for Gray output.");
    }
  } else {
    LOG_ERRORS << "Unsupported target channel count: " << target_c;
    throw std::runtime_error("Unsupported target channel count.");
  }

  // Resize & Pad
  if (args.needResize) {
    if (args.isEqualScale) {
      int img_w = current_cv_mat.cols;
      int img_h = current_cv_mat.rows;
      float scale = std::min((float)target_w / img_w, (float)target_h / img_h);
      int scaled_w = static_cast<int>(img_w * scale);
      int scaled_h = static_cast<int>(img_h * scale);

      ncnn::Mat temp_ncnn_mat = ncnn::Mat::from_pixels_resize(
          current_cv_mat.data, ncnn_pixel_type, current_cv_mat.cols,
          current_cv_mat.rows, scaled_w, scaled_h);

      args.leftPad = (target_w - scaled_w) / 2;
      args.topPad = (target_h - scaled_h) / 2;
      int rightPad = target_w - scaled_w - args.leftPad;
      int bottomPad = target_h - scaled_h - args.topPad;

      ncnn::copy_make_border(temp_ncnn_mat, ncnn_in, args.topPad, bottomPad,
                             args.leftPad, rightPad, ncnn::BORDER_CONSTANT,
                             (float)args.pad[0]);
      if (ncnn_in.w != target_w || ncnn_in.h != target_h) {
        LOG_WARNINGS << "Padded NCNN Mat size (" << ncnn_in.w << "x"
                     << ncnn_in.h << ") mismatch target (" << target_w << "x"
                     << target_h
                     << "). This is unexpected after ncnn::copy_make_border.";
      }

    } else {
      ncnn_in = ncnn::Mat::from_pixels_resize(
          current_cv_mat.data, ncnn_pixel_type, current_cv_mat.cols,
          current_cv_mat.rows, target_w, target_h);
    }
  } else {
    if (current_cv_mat.cols != target_w || current_cv_mat.rows != target_h) {
      LOG_WARNINGS << "needResize is false, but image dimensions ("
                   << current_cv_mat.cols << "x" << current_cv_mat.rows
                   << ") do not match target (" << target_w << "x" << target_h
                   << "). Resizing to target dimensions as a fallback.";
      ncnn_in = ncnn::Mat::from_pixels_resize(
          current_cv_mat.data, ncnn_pixel_type, current_cv_mat.cols,
          current_cv_mat.rows, target_w, target_h);
    } else {
      ncnn_in =
          ncnn::Mat::from_pixels(current_cv_mat.data, ncnn_pixel_type,
                                 current_cv_mat.cols, current_cv_mat.rows);
    }
  }

  if (ncnn_in.empty()) {
    LOG_ERRORS << "ncnn::Mat is empty after conversion/resize.";
    throw std::runtime_error("ncnn::Mat is empty after conversion/resize.");
  }

  // Normalization (mean subtraction and scaling)
  if (!args.meanVals.empty() || !args.normVals.empty()) {
    if (args.meanVals.size() != ncnn_in.c && !args.meanVals.empty()) {
      LOG_ERRORS << "meanVals size (" << args.meanVals.size()
                 << ") != ncnn_in.c (" << ncnn_in.c << ")";
      throw std::runtime_error(
          "meanVals size mismatch with NCNN Mat channels.");
    }
    if (args.normVals.size() != ncnn_in.c && !args.normVals.empty()) {
      LOG_ERRORS << "normVals size (" << args.normVals.size()
                 << ") != ncnn_in.c (" << ncnn_in.c << ")";
      throw std::runtime_error(
          "normVals size mismatch with NCNN Mat channels.");
    }

    std::vector<float> ncnn_norm_vals = args.normVals;
    if (!ncnn_norm_vals.empty()) {
      std::transform(ncnn_norm_vals.begin(), ncnn_norm_vals.end(),
                     ncnn_norm_vals.begin(),
                     [](float val) { return val == 0.0f ? 1.0f : 1.0f / val; });
    } else {
      if (!args.meanVals.empty()) {
        ncnn_norm_vals.assign(ncnn_in.c, 1.0f);
      }
    }

    std::vector<float> ncnn_mean_vals = args.meanVals;
    if (ncnn_mean_vals.empty() && !ncnn_norm_vals.empty()) {
      ncnn_mean_vals.assign(ncnn_in.c, 0.0f);
    }

    ncnn_in.substract_mean_normalize(
        ncnn_mean_vals.empty() ? nullptr : ncnn_mean_vals.data(),
        ncnn_norm_vals.empty() ? nullptr : ncnn_norm_vals.data());
  }

  TypedBuffer output_buffer;
  output_buffer.dataType = DataType::FLOAT32;
  output_buffer.elementCount = ncnn_in.total();

  size_t byte_size = output_buffer.elementCount * sizeof(float);
  output_buffer.data.resize(byte_size);

  if (ncnn_in.elemsize == sizeof(float)) {
    std::memcpy(output_buffer.data.data(), (unsigned char *)ncnn_in.data,
                byte_size);
  } else {
    LOG_ERRORS << "NCNN Mat elemsize is not float after processing. This is "
                  "unexpected.";
    throw std::runtime_error(
        "NCNN Mat elemsize is not float after processing.");
  }

  return output_buffer;
}

}; // namespace ai_core::dnn::mncnn
