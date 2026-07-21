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
#include "ai_core/opencv_interop.hpp"
#include "cpu_generic_preprocessor.hpp"
#include "frame_preprocessor_base.hpp"
#include <opencv2/opencv.hpp>

namespace ai_core::dnn {

namespace {
// Crop the ROI, rasterize the mask regions into an extra channel and return
// the merged image.
cv::Mat buildImageWithMaskChannel(const FrameInputWithMask &input_with_mask) {
  const auto &frame_input = input_with_mask.frame_input;

  const cv::Mat image = interop::matFromView(frame_input.image);
  const cv::Rect roi = frame_input.roi ? interop::toCv(*frame_input.roi)
                                       : cv::Rect(0, 0, image.cols, image.rows);

  cv::Mat roi_image = image(roi);
  cv::Mat mask = cv::Mat::zeros(roi_image.size(), CV_8UC1);

  for (const auto &region : input_with_mask.mask_regions) {
    cv::Rect intersection = interop::toCv(region) & roi;
    if (intersection.width <= 0 || intersection.height <= 0) {
      continue;
    }
    cv::Rect roi_space_rect(intersection.x - roi.x, intersection.y - roi.y,
                            intersection.width, intersection.height);
    cv::rectangle(mask, roi_space_rect, cv::Scalar(255), cv::FILLED);
  }

  std::vector<cv::Mat> channels(roi_image.channels());
  cv::split(roi_image, channels);
  channels.push_back(mask);

  cv::Mat image_with_mask;
  cv::merge(channels, image_with_mask);
  return image_with_mask;
}

// The mask becomes an extra channel, so mean/norm need one more entry.
FramePreprocessArg extendParamsForMaskChannel(const FramePreprocessArg &args) {
  FramePreprocessArg new_params = args;
  if (!new_params.mean_vals.empty()) {
    new_params.mean_vals.push_back(new_params.mean_vals.back());
  }
  if (!new_params.norm_vals.empty()) {
    new_params.norm_vals.push_back(new_params.norm_vals.back());
  }
  return new_params;
}
} // namespace

InferErrorCode FrameWithMaskPreprocess::process(
    const AlgoInput &input, const AlgoPreprocParams &params, TensorData &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S
        << "Failed to get FrameWithMaskPreprocArg from AlgoPreprocParams.";
    return InferErrorCode::InferInvalidInput;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "FrameWithMaskPreprocess expects exactly one input name.";
    return InferErrorCode::InferInvalidInput;
  }

  auto frame_input_with_mask = input.getParams<FrameInputWithMask>();
  if (!frame_input_with_mask) {
    LOG_ERROR_S << "Failed to get FrameInputWithMask from AlgoInput.";
    return InferErrorCode::InferInvalidInput;
  }

  const auto &frame_input = frame_input_with_mask->frame_input;
  if (frame_input.image.empty()) {
    LOG_ERROR_S << "Input frame is empty.";
    return InferErrorCode::InferInvalidInput;
  }

  FramePreprocessArg new_params = extendParamsForMaskChannel(*params_ptr);

  // ROI has already been applied while building the masked image, so the
  // masked frame carries no ROI. The Mat must outlive the view.
  cv::Mat masked_image = buildImageWithMaskChannel(*frame_input_with_mask);
  FrameInput masked_frame_input;
  masked_frame_input.image = interop::viewFromMat(masked_image);

  cpu::CpuGenericCvPreprocessor processor;
  FrameTransformContext single_runtime_args;
  TypedBuffer processed_frame =
      processor.process(new_params, masked_frame_input, single_runtime_args);

  single_runtime_args.model_input_shape = params_ptr->model_input_shape;
  single_runtime_args.is_equal_scale = params_ptr->is_equal_scale;
  single_runtime_args.roi = frame_input.roi.value_or(Rect{});

  runtime_context->frame_transform = single_runtime_args;

  std::vector<int> shape;
  if (params_ptr->hwc2chw) {
    shape = {1, params_ptr->model_input_shape.c,
             params_ptr->model_input_shape.h, params_ptr->model_input_shape.w};
  } else {
    shape = {1, params_ptr->model_input_shape.h,
             params_ptr->model_input_shape.w, params_ptr->model_input_shape.c};
  }
  output.set(params_ptr->input_names[0], std::move(processed_frame),
             std::move(shape));
  return InferErrorCode::SUCCESS;
}

InferErrorCode FrameWithMaskPreprocess::batchProcess(
    const std::vector<AlgoInput> &input, const AlgoPreprocParams &params,
    TensorData &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return InferErrorCode::InferInvalidInput;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "FrameWithMaskPreprocess expects exactly one input name.";
    return InferErrorCode::InferInvalidInput;
  }

  std::vector<FrameInput> masked_frame_inputs;
  masked_frame_inputs.reserve(input.size());
  // Owning storage for the composited images; the FrameInput views point in
  // here and must not outlive it.
  std::vector<cv::Mat> masked_images;
  masked_images.reserve(input.size());

  for (const auto &algo_input : input) {
    auto frame_input_with_mask = algo_input.getParams<FrameInputWithMask>();
    if (!frame_input_with_mask) {
      LOG_ERROR_S << "Unsupported AlgoInput type for FrameWithMaskPreprocess.";
      return InferErrorCode::InferInvalidInput;
    }

    if (frame_input_with_mask->frame_input.image.empty()) {
      LOG_ERROR_S << "Input frame is empty in the batch.";
      return InferErrorCode::InferInvalidInput;
    }

    masked_images.push_back(buildImageWithMaskChannel(*frame_input_with_mask));
    FrameInput current_masked_input;
    current_masked_input.image = interop::viewFromMat(masked_images.back());
    masked_frame_inputs.push_back(current_masked_input);
  }

  FramePreprocessArg new_params = extendParamsForMaskChannel(*params_ptr);

  cpu::CpuGenericCvPreprocessor processor;
  std::vector<FrameTransformContext> batch_runtime_args(input.size());
  TypedBuffer processed_frames = processor.batchProcess(
      new_params, masked_frame_inputs, batch_runtime_args);
  for (size_t i = 0; i < batch_runtime_args.size(); ++i) {
    batch_runtime_args[i].model_input_shape = params_ptr->model_input_shape;
    batch_runtime_args[i].is_equal_scale = params_ptr->is_equal_scale;
    batch_runtime_args[i].roi =
        input[i].getParams<FrameInputWithMask>()->frame_input.roi.value_or(
            Rect{});
  }
  runtime_context->frame_transform_batch = batch_runtime_args;

  std::vector<int> shape;
  if (params_ptr->hwc2chw) {
    shape = {static_cast<int>(input.size()), params_ptr->model_input_shape.c,
             params_ptr->model_input_shape.h, params_ptr->model_input_shape.w};
  } else {
    shape = {static_cast<int>(input.size()), params_ptr->model_input_shape.h,
             params_ptr->model_input_shape.w, params_ptr->model_input_shape.c};
  }
  output.set(params_ptr->input_names[0], std::move(processed_frames),
             std::move(shape));

  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
