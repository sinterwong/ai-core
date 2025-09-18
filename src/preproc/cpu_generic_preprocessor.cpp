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
CpuGenericCvPreprocessor::process(const FramePreprocessArg &params,
                                  const FrameInput &frameInput,
                                  FrameTransformContext &runtimeArgs) const {

  if (params.outputLocation != BufferLocation::CPU) {
    LOG_WARNINGS
        << "CPU CpuGenericCvPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }

  cv::Mat normalizedImage =
      preprocessSingleFrame(params, frameInput, runtimeArgs);

  const int inputChannels = normalizedImage.channels();

  switch (params.dataType) {
  case DataType::FLOAT32:
    return preprocessFP32(normalizedImage, inputChannels,
                          params.modelInputShape.h, params.modelInputShape.w,
                          params.hwc2chw);
  case DataType::FLOAT16:
    return preprocessFP16(normalizedImage, inputChannels,
                          params.modelInputShape.h, params.modelInputShape.w,
                          params.hwc2chw);
  default:
    LOG_ERRORS << "Unsupported data type: "
               << static_cast<int>(params.dataType);
    throw std::runtime_error("Unsupported data type");
  }
}

TypedBuffer CpuGenericCvPreprocessor::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtimeArgs) const {
  if (args.outputLocation != BufferLocation::CPU) {
    LOG_WARNINGS
        << "CPU CpuGenericCvPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }

  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batchSize = frames.size();
  runtimeArgs.resize(batchSize);

  std::vector<cv::Mat> processedImages;
  processedImages.reserve(batchSize);
  for (size_t i = 0; i < batchSize; ++i) {
    runtimeArgs[i].modelInputShape = args.modelInputShape;
    runtimeArgs[i].isEqualScale = args.isEqualScale;
    processedImages.push_back(
        preprocessSingleFrame(args, frames[i], runtimeArgs[i]));
  }

  const int inputChannels = args.modelInputShape.c;
  const int inputHeight = args.modelInputShape.h;
  const int inputWidth = args.modelInputShape.w;
  const size_t singleImageSize =
      static_cast<size_t>(inputChannels) * inputHeight * inputWidth;
  const size_t totalElements = singleImageSize * batchSize;

  // 合并所有处理后的图像到一个缓冲区中
  switch (args.dataType) {
  case DataType::FLOAT32: {
    TypedBuffer result;
    result.resize(totalElements);
    float *dstPtr = result.getHostPtr<float>();

    for (const auto &image : processedImages) {
      convertLayout(image, dstPtr, args.hwc2chw);
      dstPtr += singleImageSize; // 移动指针到下一个图像的位置
    }
    return result;
  }
  case DataType::FLOAT16: {
    std::vector<float> batchDataFP32(totalElements);
    float *dstPtr = batchDataFP32.data();

    for (const auto &image : processedImages) {
      convertLayout(image, dstPtr, args.hwc2chw);
      dstPtr += singleImageSize;
    }

    const float fp16MaxValue = 65504.0f;
    for (float &val : batchDataFP32) {
      val = std::clamp(val, -fp16MaxValue, fp16MaxValue);
    }

    cv::Mat floatMat(1, static_cast<int>(totalElements), CV_32F,
                     batchDataFP32.data());
    cv::Mat halfMat;
    floatMat.convertTo(halfMat, CV_16F);

    const size_t byteSize = totalElements * sizeof(uint16_t);
    const uint8_t *startPtr = halfMat.data;
    std::vector<uint8_t> finalData(startPtr, startPtr + byteSize);

    return TypedBuffer::createFromCpu(DataType::FLOAT16, std::move(finalData));
  }
  default:
    LOG_ERRORS << "Unsupported data type: " << static_cast<int>(args.dataType);
    throw std::runtime_error("Unsupported data type");
  }
}

cv::Mat CpuGenericCvPreprocessor::preprocessSingleFrame(
    const FramePreprocessArg &params, const FrameInput &frameInput,
    FrameTransformContext &runtimeArgs) const {
  if (frameInput.image == nullptr) {
    LOG_ERRORS << "Input frame is null.";
    throw std::runtime_error("Input frame is null.");
  }

  if (frameInput.inputRoi == nullptr) {
    runtimeArgs.roi = std::make_shared<cv::Rect>(0, 0, frameInput.image->cols,
                                                 frameInput.image->rows);
  } else {
    runtimeArgs.roi = frameInput.inputRoi;
  }
  runtimeArgs.originShape = {frameInput.image->cols, frameInput.image->rows,
                             frameInput.image->channels()};

  const auto &image = *frameInput.image;
  const auto &roi = *runtimeArgs.roi;
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    LOG_ERRORS << "Invalid ROI: " << roi << " for image size: " << image.size();
    throw std::runtime_error("Invalid ROI.");
  }

  cv::Mat croppedImage;
  if (roi.area() > 0) {
    croppedImage = image(roi).clone();
  } else {
    croppedImage = image.clone();
  }

  cv::Mat resizedImage;
  if (params.needResize) {
    if (params.isEqualScale) {
      if (params.pad.size() > 4) {
        LOG_WARNINGS << "Padding vector has more than 4 elements. Only the "
                        "first 4 will be used.";
      }
      cv::Scalar pad = utils::createScalarFromVector(params.pad);
      auto padRet = utils::escaleResizeWithPad(croppedImage, resizedImage,
                                               params.modelInputShape.h,
                                               params.modelInputShape.w, pad);
      runtimeArgs.topPad = padRet.h;
      runtimeArgs.leftPad = padRet.w;
    } else {
      cv::resize(croppedImage, resizedImage,
                 cv::Size(params.modelInputShape.w, params.modelInputShape.h),
                 0, 0, cv::INTER_LINEAR);
    }
  } else {
    resizedImage = croppedImage;
  }

  cv::Mat floatImage;
  resizedImage.convertTo(floatImage, CV_32F);

  cv::Mat normalizedImage = floatImage;
  if (!params.meanVals.empty() && !params.normVals.empty()) {
    const int inputChannels = normalizedImage.channels();
    if (params.meanVals.size() != inputChannels ||
        params.normVals.size() != inputChannels) {
      throw std::runtime_error(
          "meanVals and normVals size must match input channels");
    }

    std::vector<cv::Mat> channels(inputChannels);
    cv::split(normalizedImage, channels);

    for (int i = 0; i < inputChannels; ++i) {
      channels[i] = (channels[i] - params.meanVals[i]) / params.normVals[i];
    }
    cv::merge(channels, normalizedImage);
  }

  return normalizedImage;
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
