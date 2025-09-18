/**
 * @file frame_with_mask_prep.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "frame_with_mask_prep.hpp"
#include "ai_core/algo_data_types.hpp"
#include "cpu_generic_preprocessor.hpp"
#include "frame_preprocessor_base.hpp"
#include <logger.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>

#ifdef WITH_NCNN
#endif

#ifdef WITH_TRT
#endif

namespace ai_core::dnn {

bool FrameWithMaskPreprocess::process(
    const AlgoInput &input, const AlgoPreprocParams &params, TensorData &output,
    std::shared_ptr<RuntimeContext> &runtimeContext) const {

  auto paramsPtr = params.getParams<FramePreprocessArg>();
  if (paramsPtr == nullptr) {
    LOG_ERRORS
        << "Failed to get FrameWithMaskPreprocArg from AlgoPreprocParams.";
    return false;
  }

  if (paramsPtr->inputNames.size() != 1) {
    LOG_ERRORS << "FrameWithMaskPreprocess expects exactly one input name.";
    return false;
  }

  auto frameInputWithMask = input.getParams<FrameInputWithMask>();
  if (!frameInputWithMask) {
    LOG_ERRORS << "Failed to get FrameInputWithMask from AlgoInput.";
    return false;
  }

  const auto &frameInput = frameInputWithMask->frameInput;

  if (frameInput.image == nullptr) {
    LOG_ERRORS << "Input frame is null.";
    throw std::runtime_error("Input frame is null.");
  }

  switch (paramsPtr->preprocTaskType) {
  case FramePreprocessArg::FramePreprocType::OPENCV_CPU_CONCAT_MASK: {
    FramePreprocessArg newParams = *paramsPtr;
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();
    cv::Rect roi;
    if (frameInput.inputRoi == nullptr) {
      roi = cv::Rect(0, 0, frameInput.image->cols, frameInput.image->rows);
    } else {
      roi = *frameInput.inputRoi;
    }

    cv::Mat roiImage = (*frameInput.image)(roi);
    cv::Mat mask = cv::Mat::zeros(roiImage.size(), CV_8UC1);

    const auto &maskRegions = frameInputWithMask->maskRegions;
    for (const auto &region : maskRegions) {
      cv::Rect intersection = region & roi;
      if (intersection.width <= 0 || intersection.height <= 0) {
        continue;
      }
      cv::Rect roiSpaceRect(intersection.x - roi.x, intersection.y - roi.y,
                            intersection.width, intersection.height);
      cv::rectangle(mask, roiSpaceRect, cv::Scalar(255), cv::FILLED);
    }

    // split roi image and merge results with mask
    std::vector<cv::Mat> channels(roiImage.channels());
    cv::split(roiImage, channels);
    channels.push_back(mask);

    cv::Mat imageWithMask;
    cv::merge(channels, imageWithMask);

    // append mean and std if need
    auto &meanVals = newParams.meanVals;
    auto &normVals = newParams.normVals;

    if (!meanVals.empty()) {
      // mean for the new mask channel
      meanVals.push_back(meanVals.at(meanVals.size() - 1));
    }
    if (!normVals.empty()) {
      // norm for the new mask channel
      normVals.push_back(normVals.at(normVals.size() - 1));
    }

    FrameInput maskedFrameInput;
    maskedFrameInput.image = std::make_shared<cv::Mat>(imageWithMask);
    // empty roi(not use)
    maskedFrameInput.inputRoi = nullptr;

    // // FIXME: draw the mask to roiImage and vis
    // cv::Mat debugImage;
    // cv::cvtColor(roiImage, debugImage, cv::COLOR_BGR2BGRA);
    // for (int y = 0; y < mask.rows; ++y) {
    //   for (int x = 0; x < mask.cols; ++x) {
    //     if (mask.at<uchar>(y, x) > 0) {
    //       // Set alpha channel to 128 for masked regions
    //       debugImage.at<cv::Vec4b>(y, x)[3] = 128;
    //     }
    //   }
    // }
    // cv::imwrite("debug_masked_image.png", debugImage);

    FrameTransformContext singleRuntimeArgs;
    TypedBuffer processedFrame =
        processor->process(newParams, maskedFrameInput, singleRuntimeArgs);
    runtimeContext->setParam("preproc_runtime_args", singleRuntimeArgs);

    output.datas.insert(
        std::make_pair(paramsPtr->inputNames[0], processedFrame));

    std::vector<int> shape;
    if (paramsPtr->hwc2chw) {
      shape = {1, paramsPtr->modelInputShape.c, paramsPtr->modelInputShape.h,
               paramsPtr->modelInputShape.w};
    } else {
      shape = {1, paramsPtr->modelInputShape.h, paramsPtr->modelInputShape.w,
               paramsPtr->modelInputShape.c};
    }
    output.shapes.insert(std::make_pair(paramsPtr->inputNames[0], shape));
    break;
  }
  default: {
    LOG_ERRORS << "Unknown preprocessor type: "
               << static_cast<int>(paramsPtr->preprocTaskType);
    return false;
  }
  }
  return true;
}
} // namespace ai_core::dnn
