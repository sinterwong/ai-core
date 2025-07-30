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
#include "vision_util.hpp"
#include <logger.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

namespace ai_core::dnn::cpu {
TypedBuffer
CpuGenericCvPreprocessor::process(FramePreprocessArg &params_,
                                  const FrameInput &frameInput) const {

  if (params_.outputLocation != BufferLocation::CPU) {
    LOG_WARNINGS
        << "CPU CpuGenericCvPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }
  const auto &image = *frameInput.image;
  const auto &roi = *params_.roi;

  int inputChannels = image.channels();

  // Crop ROI
  cv::Mat croppedImage;
  if (roi.area() > 0) {
    croppedImage = image(roi).clone();
  } else {
    croppedImage = image;
  }

  // Resize
  cv::Mat resizedImage;
  if (params_.needResize) {
    if (params_.isEqualScale) {
      cv::Scalar pad;
      if (params_.pad.empty()) {
        if (inputChannels == 1) {
          pad = cv::Scalar(0);
        } else {
          pad = cv::Scalar(0, 0, 0);
        }
      } else {
        if (params_.pad.size() == 3) {
          pad = cv::Scalar(params_.pad[0], params_.pad[1], params_.pad[2]);
        } else if (params_.pad.size() == 1) {
          pad = cv::Scalar(params_.pad[0]);
        } else {
          throw std::runtime_error("Invalid pad size. Must be 1 or 3.");
        }
      }
      auto padRet = utils::escaleResizeWithPad(croppedImage, resizedImage,
                                               params_.modelInputShape.h,
                                               params_.modelInputShape.w, pad);
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

void CpuGenericCvPreprocessor::convertLayout(const cv::Mat &image, float *dst,
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

TypedBuffer
CpuGenericCvPreprocessor::preprocessFP32(const cv::Mat &normalizedImage,
                                         int inputChannels, int inputHeight,
                                         int inputWidth, bool hwc2chw) const {
  TypedBuffer result;

  const size_t totalElements =
      static_cast<size_t>(inputChannels) * inputHeight * inputWidth;
  result.resize(totalElements);

  float *dataPtr = result.getHostPtr<float>();

  convertLayout(normalizedImage, dataPtr, hwc2chw);

  return result;
}

TypedBuffer
CpuGenericCvPreprocessor::preprocessFP16(const cv::Mat &normalizedImage,
                                         int inputChannels, int inputHeight,
                                         int inputWidth, bool hwc2chw) const {
  const size_t totalElements =
      static_cast<size_t>(inputChannels) * inputHeight * inputWidth;

  std::vector<float> tensorDataFP32(totalElements);
  convertLayout(normalizedImage, tensorDataFP32.data(), hwc2chw);

  const float fp16MaxValue = 65504.0f;
  for (float &val : tensorDataFP32) {
    val = std::clamp(val, -fp16MaxValue, fp16MaxValue);
  }

  cv::Mat floatMat(1, static_cast<int>(totalElements), CV_32F,
                   tensorDataFP32.data());
  cv::Mat halfMat;
  floatMat.convertTo(halfMat, CV_16F);

  const size_t byteSize = totalElements * sizeof(uint16_t);
  const uint8_t *startPtr = halfMat.data;
  std::vector<uint8_t> finalData(startPtr, startPtr + byteSize);

  return TypedBuffer::createFromCpu(DataType::FLOAT16, std::move(finalData));
}
} // namespace ai_core::dnn::cpu
