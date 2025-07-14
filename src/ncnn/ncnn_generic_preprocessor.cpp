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

#include <algorithm>
#include <logger.hpp>
#include <ncnn/mat.h>
#include <ncnn/option.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace ai_core::dnn::mncnn {

TypedBuffer
NcnnGenericPreprocessor::process(FramePreprocessArg &args,
                                 const FrameInput &frameData) const {
  const auto &cvImageOrig = *frameData.image;
  const auto &inputRoi = *args.roi;
  if (cvImageOrig.empty()) {
    LOG_ERRORS << "Input cv::Mat image is empty.";
    throw std::runtime_error("Input cv::Mat image is empty.");
  }

  int targetWidth = args.modelInputShape.w;
  int targetHeight = args.modelInputShape.h;
  int targetChannels = cvImageOrig.channels();

  if (targetWidth <= 0 || targetHeight <= 0 || targetChannels <= 0) {
    LOG_ERRORS << "Invalid target dimensions in FramePreprocessArg: W="
               << targetWidth << ", H=" << targetHeight
               << ", C=" << targetChannels;
    throw std::runtime_error(
        "Invalid target dimensions in FramePreprocessArg.");
  }

  cv::Mat currentCvMat = cvImageOrig;

  if (inputRoi.area() > 0) {
    cv::Rect validRoi =
        inputRoi & cv::Rect(0, 0, currentCvMat.cols, currentCvMat.rows);
    if (validRoi.area() > 0) {
      currentCvMat = currentCvMat(validRoi).clone();
    } else {
      LOG_WARNINGS << "Specified ROI is outside image bounds or has zero area. "
                      "Using full image.";
    }
  }

  ncnn::Mat ncnnIn;
  int ncnnPixelType = ncnn::Mat::PIXEL_BGR;
  if (targetChannels == 3) {
    if (currentCvMat.channels() == 3) {
      ncnnPixelType = ncnn::Mat::PIXEL_BGR2RGB;
    } else if (currentCvMat.channels() == 1) {
      ncnnPixelType = ncnn::Mat::PIXEL_GRAY2RGB;
    } else {
      LOG_ERRORS << "Unsupported channel count " << currentCvMat.channels()
                 << " for 3-channel RGB output.";
      throw std::runtime_error("Unsupported channel count for RGB output.");
    }
  } else if (targetChannels == 1) {
    if (currentCvMat.channels() == 3)
      ncnnPixelType = ::ncnn::Mat::PIXEL_BGR2GRAY;
    else if (currentCvMat.channels() == 1)
      ncnnPixelType = ncnn::Mat::PIXEL_GRAY;
    else {
      LOG_ERRORS << "Unsupported channel count " << currentCvMat.channels()
                 << " for 1-channel Gray output.";
      throw std::runtime_error("Unsupported channel count for Gray output.");
    }
  } else {
    LOG_ERRORS << "Unsupported target channel count: " << targetChannels;
    throw std::runtime_error("Unsupported target channel count.");
  }

  // Resize & Pad
  if (args.needResize) {
    if (args.isEqualScale) {
      int imgWidth = currentCvMat.cols;
      int imgHeight = currentCvMat.rows;
      float scale = std::min((float)targetWidth / imgWidth,
                             (float)targetHeight / imgHeight);
      int scaledWidth = static_cast<int>(imgWidth * scale);
      int scaledHeight = static_cast<int>(imgHeight * scale);

      ncnn::Mat tempNcnnMat = ncnn::Mat::from_pixels_resize(
          currentCvMat.data, ncnnPixelType, currentCvMat.cols,
          currentCvMat.rows, scaledWidth, scaledHeight);

      args.leftPad = (targetWidth - scaledWidth) / 2;
      args.topPad = (targetHeight - scaledHeight) / 2;
      int rightPad = targetWidth - scaledWidth - args.leftPad;
      int bottomPad = targetHeight - scaledHeight - args.topPad;

      ncnn::copy_make_border(tempNcnnMat, ncnnIn, args.topPad, bottomPad,
                             args.leftPad, rightPad, ncnn::BORDER_CONSTANT,
                             (float)args.pad[0]);
      if (ncnnIn.w != targetWidth || ncnnIn.h != targetHeight) {
        LOG_WARNINGS << "Padded NCNN Mat size (" << ncnnIn.w << "x" << ncnnIn.h
                     << ") mismatch target (" << targetWidth << "x"
                     << targetHeight
                     << "). This is unexpected after ncnn::copy_make_border.";
      }

    } else {
      ncnnIn = ncnn::Mat::from_pixels_resize(
          currentCvMat.data, ncnnPixelType, currentCvMat.cols,
          currentCvMat.rows, targetWidth, targetHeight);
    }
  } else {
    if (currentCvMat.cols != targetWidth || currentCvMat.rows != targetHeight) {
      LOG_WARNINGS << "NeedResize is false, but image dimensions ("
                   << currentCvMat.cols << "x" << currentCvMat.rows
                   << ") do not match target (" << targetWidth << "x"
                   << targetHeight
                   << "). Resizing to target dimensions as a fallback.";
      ncnnIn = ncnn::Mat::from_pixels_resize(
          currentCvMat.data, ncnnPixelType, currentCvMat.cols,
          currentCvMat.rows, targetWidth, targetHeight);
    } else {
      ncnnIn = ncnn::Mat::from_pixels(currentCvMat.data, ncnnPixelType,
                                      currentCvMat.cols, currentCvMat.rows);
    }
  }

  if (ncnnIn.empty()) {
    LOG_ERRORS << "ncnn::Mat is empty after conversion/resize.";
    throw std::runtime_error("ncnn::Mat is empty after conversion/resize.");
  }

  // Normalization (mean subtraction and scaling)
  if (!args.meanVals.empty() || !args.normVals.empty()) {
    if (args.meanVals.size() != ncnnIn.c && !args.meanVals.empty()) {
      LOG_ERRORS << "meanVals size (" << args.meanVals.size()
                 << ") != ncnnIn.c (" << ncnnIn.c << ")";
      throw std::runtime_error(
          "MeanVals size mismatch with NCNN Mat channels.");
    }
    if (args.normVals.size() != ncnnIn.c && !args.normVals.empty()) {
      LOG_ERRORS << "normVals size (" << args.normVals.size()
                 << ") != ncnnIn.c (" << ncnnIn.c << ")";
      throw std::runtime_error(
          "NormVals size mismatch with NCNN Mat channels.");
    }

    std::vector<float> ncnnNormVals = args.normVals;
    if (!ncnnNormVals.empty()) {
      std::transform(ncnnNormVals.begin(), ncnnNormVals.end(),
                     ncnnNormVals.begin(),
                     [](float val) { return val == 0.0f ? 1.0f : 1.0f / val; });
    } else {
      if (!args.meanVals.empty()) {
        ncnnNormVals.assign(ncnnIn.c, 1.0f);
      }
    }

    std::vector<float> ncnnMeanVals = args.meanVals;
    if (ncnnMeanVals.empty() && !ncnnNormVals.empty()) {
      ncnnMeanVals.assign(ncnnIn.c, 0.0f);
    }

    ncnnIn.substract_mean_normalize(
        ncnnMeanVals.empty() ? nullptr : ncnnMeanVals.data(),
        ncnnNormVals.empty() ? nullptr : ncnnNormVals.data());
  }

  if (ncnnIn.elemsize != sizeof(float)) {
    LOG_ERRORS << "NCNN Mat elemsize is not float. Unexpected data format.";
    throw std::runtime_error("NCNN Mat elemsize is not float.");
  }

  const size_t byteSize = ncnnIn.total() * sizeof(float);
  const uint8_t *ncnnDataPtr = reinterpret_cast<const uint8_t *>(ncnnIn.data);

  std::vector<uint8_t> cpuDataVec(ncnnDataPtr, ncnnDataPtr + byteSize);

  TypedBuffer outputBuffer =
      TypedBuffer::createFromCpu(DataType::FLOAT32, std::move(cpuDataVec));
  return outputBuffer;
}
}; // namespace ai_core::dnn::mncnn
