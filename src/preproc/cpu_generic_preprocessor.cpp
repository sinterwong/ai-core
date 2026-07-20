/**
 * @file cpu_generic_preprocessor.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.2
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "cpu_generic_preprocessor.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include "vision_util.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <type_traits>

namespace ai_core::dnn::cpu {

namespace {

// Scalar float -> IEEE-754 half (finite, in-range values). Kept local and
// inlined for the fp16 hot path.
inline uint16_t floatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  const uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00u); // saturate to inf
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (mant >> 13));
}

// One fused pass: read the prepared uint8 image and write normalized values in
// the requested dtype/layout. Templating on Dst (float/uint16) and the layout
// flag lets the compiler drop the per-pixel branches.
template <typename Dst, bool Chw>
void fusedWrite(const cv::Mat &img, Dst *dst, const float *scale,
                const float *shift, int channels) {
  const int height = img.rows;
  const int width = img.cols;
  const size_t plane = static_cast<size_t>(height) * width;

  auto store = [](Dst *p, float v) {
    if constexpr (std::is_same_v<Dst, uint16_t>) {
      constexpr float kHalfMax = 65504.0f;
      *p = floatToHalf(std::clamp(v, -kHalfMax, kHalfMax));
    } else {
      *p = v;
    }
  };

  for (int y = 0; y < height; ++y) {
    const uint8_t *row = img.ptr<uint8_t>(y);
    for (int x = 0; x < width; ++x) {
      const uint8_t *px = row + static_cast<size_t>(x) * channels;
      for (int c = 0; c < channels; ++c) {
        const float v = static_cast<float>(px[c]) * scale[c] + shift[c];
        if constexpr (Chw) {
          store(dst + c * plane + static_cast<size_t>(y) * width + x, v);
        } else {
          store(dst + (static_cast<size_t>(y) * width + x) * channels + c, v);
        }
      }
    }
  }
}

} // namespace

TypedBuffer
CpuGenericCvPreprocessor::process(const FramePreprocessArg &params,
                                  const FrameInput &frame_input,
                                  FrameTransformContext &runtime_args) const {

  if (params.output_location != BufferLocation::CPU) {
    LOG_WARNING_S
        << "CPU CpuGenericCvPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }

  cv::Mat prepared = cropAndResize(params, frame_input, runtime_args);

  const size_t total_elements = static_cast<size_t>(prepared.channels()) *
                                params.model_input_shape.h *
                                params.model_input_shape.w;

  TypedBuffer result;
  switch (params.data_type) {
  case DataType::FLOAT32:
    result = TypedBuffer::createFromCpu(DataType::FLOAT32,
                                        std::vector<uint8_t>(total_elements *
                                                             sizeof(float)));
    break;
  case DataType::FLOAT16:
    result = TypedBuffer::createFromCpu(DataType::FLOAT16,
                                        std::vector<uint8_t>(total_elements *
                                                             sizeof(uint16_t)));
    break;
  default:
    LOG_ERROR_S << "Unsupported data type: "
                << static_cast<int>(params.data_type);
    throw std::runtime_error("Unsupported data type");
  }
  writeNormalizedLayout(prepared, params, result, 0);
  return result;
}

TypedBuffer CpuGenericCvPreprocessor::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtime_args) const {
  if (args.output_location != BufferLocation::CPU) {
    LOG_WARNING_S
        << "CPU CpuGenericCvPreprocessor requested to output to GPU_DEVICE. "
           "This is not supported. Output will be on CPU.";
  }

  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batch_size = frames.size();
  runtime_args.resize(batch_size);

  const int input_channels = args.model_input_shape.c;
  const int input_height = args.model_input_shape.h;
  const int input_width = args.model_input_shape.w;
  const size_t single_image_size =
      static_cast<size_t>(input_channels) * input_height * input_width;
  const size_t total_elements = single_image_size * batch_size;

  const size_t elem_size = (args.data_type == DataType::FLOAT16)
                               ? sizeof(uint16_t)
                               : sizeof(float);
  if (args.data_type != DataType::FLOAT32 &&
      args.data_type != DataType::FLOAT16) {
    LOG_ERROR_S << "Unsupported data type: "
                << static_cast<int>(args.data_type);
    throw std::runtime_error("Unsupported data type");
  }

  TypedBuffer result = TypedBuffer::createFromCpu(
      args.data_type, std::vector<uint8_t>(total_elements * elem_size));

  for (size_t i = 0; i < batch_size; ++i) {
    runtime_args[i].model_input_shape = args.model_input_shape;
    runtime_args[i].is_equal_scale = args.is_equal_scale;
    cv::Mat prepared = cropAndResize(args, frames[i], runtime_args[i]);
    writeNormalizedLayout(prepared, args, result, i * single_image_size);
  }
  return result;
}

cv::Mat CpuGenericCvPreprocessor::cropAndResize(
    const FramePreprocessArg &params, const FrameInput &frame_input,
    FrameTransformContext &runtime_args) const {
  if (frame_input.image.empty()) {
    LOG_ERROR_S << "Input frame is empty.";
    throw std::runtime_error("Input frame is empty.");
  }

  const cv::Mat image = interop::matFromView(frame_input.image);

  runtime_args.roi = frame_input.roi.value_or(
      Rect{0, 0, frame_input.image.width, frame_input.image.height});
  runtime_args.origin_shape = {frame_input.image.width,
                               frame_input.image.height,
                               frame_input.image.channels()};

  const cv::Rect roi = interop::toCv(runtime_args.roi);
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    LOG_ERROR_S << "Invalid ROI: " << roi
                << " for image size: " << image.size();
    throw std::runtime_error("Invalid ROI.");
  }

  // Crop as a view (no copy); resize allocates the one working buffer we need.
  cv::Mat cropped = image(roi);

  cv::Mat resized;
  if (params.need_resize) {
    if (params.is_equal_scale) {
      if (params.pad.size() > 4) {
        LOG_WARNING_S << "Padding vector has more than 4 elements. Only the "
                         "first 4 will be used.";
      }
      cv::Scalar pad = utils::createScalarFromVector(params.pad);
      auto pad_ret = utils::escaleResizeWithPad(cropped, resized,
                                                params.model_input_shape.h,
                                                params.model_input_shape.w, pad);
      runtime_args.top_pad = pad_ret.h;
      runtime_args.left_pad = pad_ret.w;
    } else {
      cv::resize(
          cropped, resized,
          cv::Size(params.model_input_shape.w, params.model_input_shape.h), 0,
          0, cv::INTER_LINEAR);
    }
  } else {
    // No resize requested: still ensure contiguous storage for the fused pass.
    resized = cropped.isContinuous() ? cropped : cropped.clone();
  }

  return resized;
}

void CpuGenericCvPreprocessor::writeNormalizedLayout(
    const cv::Mat &prepared_u8, const FramePreprocessArg &params,
    TypedBuffer &dst, size_t dst_offset_elems) const {
  const int channels = prepared_u8.channels();

  if (!params.mean_vals.empty() &&
      (params.mean_vals.size() != static_cast<size_t>(channels) ||
       params.norm_vals.size() != static_cast<size_t>(channels))) {
    throw std::runtime_error(
        "mean_vals and norm_vals size must match input channels");
  }

  // Fold (v - mean) / norm into v * scale + shift, per channel.
  std::array<float, 4> scale{1.f, 1.f, 1.f, 1.f};
  std::array<float, 4> shift{0.f, 0.f, 0.f, 0.f};
  if (!params.mean_vals.empty()) {
    for (int c = 0; c < channels && c < 4; ++c) {
      scale[c] = 1.f / params.norm_vals[c];
      shift[c] = -params.mean_vals[c] / params.norm_vals[c];
    }
  }

  if (params.data_type == DataType::FLOAT32) {
    float *out = dst.getHostPtr<float>() + dst_offset_elems;
    if (params.hwc2chw) {
      fusedWrite<float, true>(prepared_u8, out, scale.data(), shift.data(),
                              channels);
    } else {
      fusedWrite<float, false>(prepared_u8, out, scale.data(), shift.data(),
                               channels);
    }
  } else { // FLOAT16
    uint16_t *out = dst.getHostPtr<uint16_t>() + dst_offset_elems;
    if (params.hwc2chw) {
      fusedWrite<uint16_t, true>(prepared_u8, out, scale.data(), shift.data(),
                                 channels);
    } else {
      fusedWrite<uint16_t, false>(prepared_u8, out, scale.data(), shift.data(),
                                  channels);
    }
  }
}
} // namespace ai_core::dnn::cpu
