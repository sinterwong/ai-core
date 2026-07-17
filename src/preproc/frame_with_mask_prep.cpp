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
#include "ai_core/algo_types.hpp"
#include "ai_core/logger.hpp"
#include "cpu_generic_preprocessor.hpp"
#include "frame_preprocessor_base.hpp"
#include <opencv2/opencv.hpp>
#include <ostream>

#ifdef WITH_NCNN
#endif

#ifdef WITH_TRT
#endif

namespace ai_core::dnn {

InferErrorCode FrameWithMaskPreprocess::process(
    const AlgoInput &input, const AlgoPreprocParams &params, TensorData &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S
        << "Failed to get FrameWithMaskPreprocArg from AlgoPreprocParams.";
    return InferErrorCode::InferPreprocessFailed;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "FrameWithMaskPreprocess expects exactly one input name.";
    return InferErrorCode::InferPreprocessFailed;
  }

  auto frame_input_with_mask = input.getParams<FrameInputWithMask>();
  if (!frame_input_with_mask) {
    LOG_ERROR_S << "Failed to get FrameInputWithMask from AlgoInput.";
    return InferErrorCode::InferPreprocessFailed;
  }

  const auto &frame_input = frame_input_with_mask->frame_input;

  if (frame_input.image == nullptr) {
    LOG_ERROR_S << "Input frame is null.";
    throw std::runtime_error("Input frame is null.");
  }

  switch (params_ptr->preproc_task_type) {
  case FramePreprocessArg::FramePreprocType::OpencvCpuConcatMask: {
    FramePreprocessArg new_params = *params_ptr;
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();
    cv::Rect roi;
    if (frame_input.input_roi == nullptr) {
      roi = cv::Rect(0, 0, frame_input.image->cols, frame_input.image->rows);
    } else {
      roi = *frame_input.input_roi;
    }

    cv::Mat roi_image = (*frame_input.image)(roi);
    cv::Mat mask = cv::Mat::zeros(roi_image.size(), CV_8UC1);

    const auto &mask_regions = frame_input_with_mask->mask_regions;
    for (const auto &region : mask_regions) {
      cv::Rect intersection = region & roi;
      if (intersection.width <= 0 || intersection.height <= 0) {
        continue;
      }
      cv::Rect roiSpaceRect(intersection.x - roi.x, intersection.y - roi.y,
                            intersection.width, intersection.height);
      cv::rectangle(mask, roiSpaceRect, cv::Scalar(255), cv::FILLED);
    }

    // split roi image and merge results with mask
    std::vector<cv::Mat> channels(roi_image.channels());
    cv::split(roi_image, channels);
    channels.push_back(mask);

    cv::Mat image_with_mask;
    cv::merge(channels, image_with_mask);

    // append mean and std if need
    auto &mean_vals = new_params.mean_vals;
    auto &norm_vals = new_params.norm_vals;

    if (!mean_vals.empty()) {
      // mean for the new mask channel
      mean_vals.push_back(mean_vals.at(mean_vals.size() - 1));
    }
    if (!norm_vals.empty()) {
      // norm for the new mask channel
      norm_vals.push_back(norm_vals.at(norm_vals.size() - 1));
    }

    FrameInput masked_frame_input;
    masked_frame_input.image = std::make_shared<cv::Mat>(image_with_mask);
    // empty roi(not use)
    masked_frame_input.input_roi = nullptr;

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

    FrameTransformContext single_runtime_args;
    TypedBuffer processed_frame =
        processor->process(new_params, masked_frame_input, single_runtime_args);

    single_runtime_args.model_input_shape = params_ptr->model_input_shape;
    single_runtime_args.is_equal_scale = params_ptr->is_equal_scale;
    single_runtime_args.roi = frame_input.input_roi;

    runtime_context->setParam("preproc_runtime_args", single_runtime_args);

    output.datas.insert(
        std::make_pair(params_ptr->input_names[0], processed_frame));

    std::vector<int> shape;
    if (params_ptr->hwc2chw) {
      shape = {1, params_ptr->model_input_shape.c,
               params_ptr->model_input_shape.h,
               params_ptr->model_input_shape.w};
    } else {
      shape = {1, params_ptr->model_input_shape.h,
               params_ptr->model_input_shape.w,
               params_ptr->model_input_shape.c};
    }
    output.shapes.insert(std::make_pair(params_ptr->input_names[0], shape));
    break;
  }
  default: {
    LOG_ERROR_S << "Unknown preprocessor type: "
                << static_cast<int>(params_ptr->preproc_task_type);
    return InferErrorCode::InferPreprocessFailed;
  }
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode FrameWithMaskPreprocess::batchProcess(
    const std::vector<AlgoInput> &input, const AlgoPreprocParams &params,
    TensorData &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return InferErrorCode::InferPreprocessFailed;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "FrameWithMaskPreprocess expects exactly one input name.";
    return InferErrorCode::InferPreprocessFailed;
  }

  std::vector<FrameInput> masked_frame_inputs;
  masked_frame_inputs.reserve(input.size());

  for (const auto &algo_input : input) {
    auto frame_input_with_mask = algo_input.getParams<FrameInputWithMask>();
    if (!frame_input_with_mask) {
      LOG_ERROR_S << "Unsupported AlgoInput type for FrameWithMaskPreprocess.";
      return InferErrorCode::InferPreprocessFailed;
    }
    const auto &frame_input = frame_input_with_mask->frame_input;

    if (frame_input.image == nullptr) {
      LOG_ERROR_S << "Input frame is null in the batch.";
      return InferErrorCode::InferPreprocessFailed;
    }

    cv::Rect roi;
    if (frame_input.input_roi == nullptr) {
      roi = cv::Rect(0, 0, frame_input.image->cols, frame_input.image->rows);
    } else {
      roi = *frame_input.input_roi;
    }

    cv::Mat roiImage = (*frame_input.image)(roi);
    cv::Mat mask = cv::Mat::zeros(roiImage.size(), CV_8UC1);

    const auto &mask_regions = frame_input_with_mask->mask_regions;
    for (const auto &region : mask_regions) {
      cv::Rect intersection = region & roi;
      if (intersection.width <= 0 || intersection.height <= 0) {
        continue;
      }
      cv::Rect roiSpaceRect(intersection.x - roi.x, intersection.y - roi.y,
                            intersection.width, intersection.height);
      cv::rectangle(mask, roiSpaceRect, cv::Scalar(255), cv::FILLED);
    }

    std::vector<cv::Mat> channels(roiImage.channels());
    cv::split(roiImage, channels);
    channels.push_back(mask);

    cv::Mat imageWithMask;
    cv::merge(channels, imageWithMask);

    FrameInput current_masked_input;
    current_masked_input.image = std::make_shared<cv::Mat>(imageWithMask);
    current_masked_input.input_roi = nullptr;
    masked_frame_inputs.push_back(current_masked_input);
  }

  switch (params_ptr->preproc_task_type) {
  case FramePreprocessArg::FramePreprocType::OpencvCpuConcatMask: {
    FramePreprocessArg new_params = *params_ptr;
    auto &mean_vals = new_params.mean_vals;
    auto &norm_vals = new_params.norm_vals;

    if (!mean_vals.empty()) {
      mean_vals.push_back(mean_vals.back());
    }
    if (!norm_vals.empty()) {
      norm_vals.push_back(norm_vals.back());
    }

    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();

    std::vector<FrameTransformContext> batch_runtime_args(input.size());
    TypedBuffer processed_frames = processor->batchProcess(
        new_params, masked_frame_inputs, batch_runtime_args);
    for (size_t i = 0; i < batch_runtime_args.size(); ++i) {
      batch_runtime_args[i].model_input_shape = params_ptr->model_input_shape;
      batch_runtime_args[i].is_equal_scale = params_ptr->is_equal_scale;
      batch_runtime_args[i].roi =
          input[i].getParams<FrameInputWithMask>()->frame_input.input_roi;
    }
    runtime_context->setParam("preproc_runtime_args_batch", batch_runtime_args);

    output.datas.insert(
        std::make_pair(params_ptr->input_names[0], processed_frames));

    break;
  }
  default: {
    LOG_ERROR_S << "Unknown or unsupported batch preprocessor type for "
                   "FrameWithMask: "
                << static_cast<int>(params_ptr->preproc_task_type);
    return InferErrorCode::InferPreprocessFailed;
  }
  }

  std::vector<int> shape;
  if (params_ptr->hwc2chw) {
    shape = {static_cast<int>(input.size()), params_ptr->model_input_shape.c,
             params_ptr->model_input_shape.h, params_ptr->model_input_shape.w};
  } else {
    shape = {static_cast<int>(input.size()), params_ptr->model_input_shape.h,
             params_ptr->model_input_shape.w, params_ptr->model_input_shape.c};
  }
  output.shapes.insert(std::make_pair(params_ptr->input_names[0], shape));

  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
