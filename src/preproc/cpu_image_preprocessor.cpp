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
#include "cpu_image_preprocessor.hpp"
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
                          params_.modelInputShape.h, params_.modelInputShape.w);

  case DataType::FLOAT16:
    return preprocessFP16(normalizedImage, inputChannels,
                          params_.modelInputShape.h, params_.modelInputShape.w);

  default:
    LOG_ERRORS << "Unsupported data type: "
               << static_cast<int>(params_.dataType);
    throw std::runtime_error("Unsupported data type");
  }
}

TypedBuffer ImagePreprocessor::preprocessFP32(const cv::Mat &normalizedImage,
                                              int inputChannels,
                                              int inputHeight,
                                              int inputWidth) const {

  TypedBuffer result;
  result.dataType = DataType::FLOAT32;

  std::vector<float> tensorData;
  tensorData.reserve(inputChannels * inputHeight * inputWidth);

  if (normalizedImage.channels() == 1) {
    if (normalizedImage.isContinuous()) {
      const float *srcData =
          reinterpret_cast<const float *>(normalizedImage.data);
      const size_t totalSize = inputWidth * inputHeight;
      tensorData.assign(srcData, srcData + totalSize);
    } else {
      for (int h = 0; h < inputHeight; ++h) {
        for (int w = 0; w < inputWidth; ++w) {
          tensorData.push_back(normalizedImage.at<float>(h, w));
        }
      }
    }
  } else if (normalizedImage.channels() == 3) {
    tensorData.resize(inputChannels * inputHeight * inputWidth);
    const int planeSize = inputHeight * inputWidth;

    for (int h = 0; h < inputHeight; ++h) {
      for (int w = 0; w < inputWidth; ++w) {
        const cv::Vec3f &pixel = normalizedImage.at<cv::Vec3f>(h, w);
        const int hwIndex = h * inputWidth + w;

        for (int c = 0; c < inputChannels; ++c) {
          tensorData[c * planeSize + hwIndex] = pixel[c];
        }
      }
    }
  } else {
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(normalizedImage.channels()));
  }

  const size_t byteSize = tensorData.size() * sizeof(float);
  std::vector<uint8_t> byteData(byteSize);
  std::memcpy(byteData.data(), tensorData.data(), byteSize);

  result.data = std::move(byteData);
  result.elementCount = tensorData.size();

  return result;
}

TypedBuffer ImagePreprocessor::preprocessFP16(const cv::Mat &normalizedImage,
                                              int inputChannels,
                                              int inputHeight,
                                              int inputWidth) const {

  TypedBuffer result;
  result.dataType = DataType::FLOAT16;

  std::vector<float> tensorDataFP32;
  tensorDataFP32.reserve(inputChannels * inputHeight * inputWidth);

  const float fp16MaxValue = 65504.0f;

  if (normalizedImage.channels() == 1) {
    if (normalizedImage.isContinuous()) {
      const float *srcData =
          reinterpret_cast<const float *>(normalizedImage.data);
      const size_t totalSize = inputWidth * inputHeight;

      tensorDataFP32.resize(totalSize);
      for (size_t i = 0; i < totalSize; ++i) {
        tensorDataFP32[i] = std::clamp(srcData[i], -fp16MaxValue, fp16MaxValue);
      }
    } else {
      for (int h = 0; h < inputHeight; ++h) {
        for (int w = 0; w < inputWidth; ++w) {
          float val = normalizedImage.at<float>(h, w);
          tensorDataFP32.push_back(
              std::clamp(val, -fp16MaxValue, fp16MaxValue));
        }
      }
    }
  } else if (normalizedImage.channels() == 3) {
    tensorDataFP32.resize(inputChannels * inputHeight * inputWidth);
    const int planeSize = inputHeight * inputWidth;

    for (int h = 0; h < inputHeight; ++h) {
      for (int w = 0; w < inputWidth; ++w) {
        const cv::Vec3f &pixel = normalizedImage.at<cv::Vec3f>(h, w);
        const int hwIndex = h * inputWidth + w;

        for (int c = 0; c < inputChannels; ++c) {
          tensorDataFP32[c * planeSize + hwIndex] =
              std::clamp(pixel[c], -fp16MaxValue, fp16MaxValue);
        }
      }
    }
  } else {
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(normalizedImage.channels()));
  }

  cv::Mat floatMat(1, tensorDataFP32.size(), CV_32F, tensorDataFP32.data());
  cv::Mat halfMat(1, tensorDataFP32.size(), CV_16F);
  floatMat.convertTo(halfMat, CV_16F);

  const size_t byteSize = tensorDataFP32.size() * sizeof(uint16_t);
  std::vector<uint8_t> byteData(byteSize);
  std::memcpy(byteData.data(), halfMat.data, byteSize);

  result.data = std::move(byteData);
  result.elementCount = tensorDataFP32.size();

  return result;
}
} // namespace ai_core::dnn::cpu