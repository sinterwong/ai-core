/**
 * @file cpu_image_preprocessor.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "opencv_image_preprocessor.hpp"
#include "logger.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "vision_util.hpp"

namespace ai_core::dnn::cpu {
TypedBuffer ImagePreprocessor::process(FramePreprocessArg &params_,
                                       const FrameInput &frameInput) const {
  const cv::Mat &image = frameInput.image;
  int inputChannels = image.channels();

  // Crop ROI
  cv::Mat croppedImage;
  if (params_.roi.area() > 0) {
    croppedImage = image(params_.roi).clone();
  } else {
    croppedImage = image;
  }

  // Resize
  cv::Mat resizedImage;
  if (params_.needResize) {
    if (params_.isEqualScale) {
      auto padRet = utils::escaleResizeWithPad(
          croppedImage, resizedImage, params_.modelInputShape.h,
          params_.modelInputShape.w, params_.pad);
      params_.topPad = padRet.h;
      params_.leftPad = padRet.w;
    } else {
      cv::resize(croppedImage, resizedImage,
                 cv::Size(params_.modelInputShape.w, params_.modelInputShape.h),
                 0, 0, cv::INTER_LINEAR);
    }
  } else {
    resizedImage = croppedImage;
  }

  // Convert to float
  cv::Mat floatImage;
  resizedImage.convertTo(floatImage, CV_32F);

  // Normalization
  cv::Mat normalizedImage;
  if (!params_.meanVals.empty() && !params_.normVals.empty()) {
    // Validate normalization parameters
    if (params_.meanVals.size() != inputChannels ||
        params_.normVals.size() != inputChannels) {
      throw std::runtime_error(
          "meanVals and normVals size must match input channels");
    }

    std::vector<cv::Mat> channels(inputChannels);
    cv::split(floatImage, channels);

    // Apply normalization per channel
    for (int i = 0; i < inputChannels; ++i) {
      channels[i] = (channels[i] - params_.meanVals[i]) / params_.normVals[i];
    }
    cv::merge(channels, normalizedImage);
  } else {
    normalizedImage = floatImage;
  }

  switch (params_.dataType) {
  case DataType::FLOAT32:
    return preprocessFP32(normalizedImage, inputChannels,
                          params_.modelInputShape.h, params_.modelInputShape.w,
                          params_.hwc2chw);
  case DataType::FLOAT16:
    return preprocessFP16(normalizedImage, inputChannels,
                          params_.modelInputShape.h, params_.modelInputShape.w,
                          params_.hwc2chw);

  default:
    LOG_ERRORS << "Unsupported data type: "
               << static_cast<int>(params_.dataType);
    throw std::runtime_error("Unsupported data type");
  }
}

void ImagePreprocessor::convertLayout(const cv::Mat &image, float *dst,
                                      bool hwc2chw) const {
  const int height = image.rows;
  const int width = image.cols;
  const int channels = image.channels();
  const size_t totalElements = static_cast<size_t>(height) * width * channels;

  if (!hwc2chw || channels == 1) {
    if (image.isContinuous()) {
      std::memcpy(dst, image.data, totalElements * sizeof(float));
    } else {
      const size_t rowSize = width * channels * sizeof(float);
      for (int h = 0; h < height; ++h) {
        std::memcpy(dst + h * width * channels, image.ptr<float>(h), rowSize);
      }
    }
  } else {
    const int planeSize = height * width;
    for (int h = 0; h < height; ++h) {
      const float *rowPtr = image.ptr<float>(h);
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          dst[c * planeSize + h * width + w] = rowPtr[w * channels + c];
        }
      }
    }
  }
}

TypedBuffer ImagePreprocessor::preprocessFP32(const cv::Mat &normalizedImage,
                                              int inputChannels,
                                              int inputHeight, int inputWidth,
                                              bool hwc2chw) const {
  TypedBuffer result;
  result.dataType = DataType::FLOAT32;
  const size_t totalElements =
      static_cast<size_t>(inputChannels) * inputHeight * inputWidth;

  result.data.resize(totalElements * sizeof(float));

  convertLayout(normalizedImage, reinterpret_cast<float *>(result.data.data()),
                hwc2chw);

  result.elementCount = totalElements;
  return result;
}

TypedBuffer ImagePreprocessor::preprocessFP16(const cv::Mat &normalizedImage,
                                              int inputChannels,
                                              int inputHeight, int inputWidth,
                                              bool hwc2chw) const {
  TypedBuffer result;
  result.dataType = DataType::FLOAT16;
  const size_t totalElements =
      static_cast<size_t>(inputChannels) * inputHeight * inputWidth;

  std::vector<float> tensorDataFP32(totalElements);

  convertLayout(normalizedImage, tensorDataFP32.data(), hwc2chw);

  const float fp16MaxValue = 65504.0f;
  for (size_t i = 0; i < totalElements; ++i) {
    tensorDataFP32[i] =
        std::clamp(tensorDataFP32[i], -fp16MaxValue, fp16MaxValue);
  }

  cv::Mat floatMat(1, totalElements, CV_32F, tensorDataFP32.data());
  cv::Mat halfMat;
  floatMat.convertTo(halfMat, CV_16F);

  result.data.resize(totalElements * sizeof(uint16_t));
  std::memcpy(result.data.data(), halfMat.data, result.data.size());

  result.elementCount = totalElements;
  return result;
}
} // namespace ai_core::dnn::cpu