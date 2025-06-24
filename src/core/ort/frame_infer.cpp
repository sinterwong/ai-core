/**
 * @file dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "frame_infer.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"
#include "vision_util.hpp"

namespace infer::dnn {

std::vector<TypedBuffer> FrameInference::preprocess(AlgoInput &input) const {
  // Get input parameters
  auto *frameInput = input.getParams<FrameInput>();
  if (!frameInput) {
    LOG_ERRORS << "Invalid input parameters";
    throw std::runtime_error("Invalid input parameters");
  }

  if (inputNames.size() != 1) {
    LOG_ERRORS << "Input node number is not 1";
    throw std::runtime_error("Input node number is not 1");
  }

  const auto &inputShape = inputShapes.at(0);
  const size_t numDims = inputShape.size();

  // Parse input dimensions
  int inputWidth = 1, inputHeight = 1, inputChannels = 1, inputBatch = 1;

  // Handle different input dimensions
  if (numDims >= 2) {
    inputWidth = inputShape[numDims - 1];
    inputHeight = inputShape[numDims - 2];
  }
  if (numDims >= 3) {
    inputChannels = inputShape[numDims - 3];
  }
  if (numDims >= 4) {
    inputBatch = inputShape[numDims - 4];
  }

  // Ensure batch size is 1
  if (inputBatch != 1) {
    throw std::runtime_error("Only batch size 1 is supported");
  }

  const cv::Mat &image = frameInput->image;
  auto &args = frameInput->args;

  // Crop ROI
  cv::Mat croppedImage;
  if (args.roi.area() > 0) {
    croppedImage = image(args.roi).clone();
  } else {
    croppedImage = image;
  }

  // Resize
  cv::Mat resizedImage;
  if (args.needResize) {
    if (args.isEqualScale) {
      auto padRet = utils::escaleResizeWithPad(
          croppedImage, resizedImage, inputHeight, inputWidth, args.pad);
      args.topPad = padRet.h;
      args.leftPad = padRet.w;
    } else {
      cv::resize(croppedImage, resizedImage, cv::Size(inputWidth, inputHeight),
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
  if (!args.meanVals.empty() && !args.normVals.empty()) {
    // Validate normalization parameters
    if (args.meanVals.size() != inputChannels ||
        args.normVals.size() != inputChannels) {
      throw std::runtime_error(
          "meanVals and normVals size must match input channels");
    }

    std::vector<cv::Mat> channels(inputChannels);
    cv::split(floatImage, channels);

    // Apply normalization per channel
    for (int i = 0; i < inputChannels; ++i) {
      channels[i] = (channels[i] - args.meanVals[i]) / args.normVals[i];
    }
    cv::merge(channels, normalizedImage);
  } else {
    normalizedImage = floatImage;
  }

  switch (params->dataType) {
  case DataType::FLOAT32:
    return preprocessFP32(normalizedImage, inputChannels, inputHeight,
                          inputWidth);

  case DataType::FLOAT16:
    return preprocessFP16(normalizedImage, inputChannels, inputHeight,
                          inputWidth);

  default:
    LOG_ERRORS << "Unsupported data type: "
               << static_cast<int>(params->dataType);
    throw std::runtime_error("Unsupported data type");
  }
}

std::vector<TypedBuffer>
FrameInference::preprocessFP32(const cv::Mat &normalizedImage,
                               int inputChannels, int inputHeight,
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

  return {result};
}

std::vector<TypedBuffer>
FrameInference::preprocessFP16(const cv::Mat &normalizedImage,
                               int inputChannels, int inputHeight,
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

  return {result};
}
} // namespace infer::dnn