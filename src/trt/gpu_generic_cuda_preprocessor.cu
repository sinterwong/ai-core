/**
 * @file gpu_generic_cuda_preprocessor.cu
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief GPU-accelerated frame preprocessor implementation
 * @version 0.2
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/logger.hpp"
#include "ai_core/typed_buffer.hpp"
#include "cuda_stream.cuh"
#include "cuda_utils.hpp"
#include "gpu_generic_cuda_preprocessor.hpp"
#include <cmath>
#include <opencv2/core.hpp>

namespace ai_core::dnn::gpu {
using namespace cuda_utils;

// ===========================================================================
// CudaStream Implementation
// ===========================================================================

void GpuGenericCudaPreprocessor::CachedResources::reset() {
  d_mean.reset();
  d_std.reset();
  d_pad.reset();
  cached_mean_vals.clear();
  cached_norm_vals.clear();
  cached_pad_vals.clear();
  d_hwc_buffer.reset();
  d_chw_buffer.reset();
  d_input_image.reset();
  d_batch_input_images.clear();
  d_src_ptrs.reset();
  d_src_heights.reset();
  d_src_widths.reset();
  d_rois.reset();
  d_new_heights.reset();
  d_new_widths.reset();
  d_pad_ys.reset();
  d_pad_xs.reset();
}

// ===========================================================================
// GpuGenericCudaPreprocessor Implementation
// ===========================================================================

GpuGenericCudaPreprocessor::GpuGenericCudaPreprocessor()
    : GpuGenericCudaPreprocessor(Config::defaults()) {}

GpuGenericCudaPreprocessor::GpuGenericCudaPreprocessor(const Config &config)
    : m_config(config) {
  // Only create stream for sequential mode
  if (!config.enable_parallel) {
    m_stream = std::make_unique<CudaStream>(
        config.use_high_priority_stream ? CudaStream::Priority::High
                                     : CudaStream::Priority::Default);
  }
}

GpuGenericCudaPreprocessor::~GpuGenericCudaPreprocessor() {
  if (m_stream) {
    try {
      m_stream->synchronize();
    } catch (...) {
    }
  }
}

cudaStream_t GpuGenericCudaPreprocessor::getStream() const {
  if (m_config.enable_parallel) {
    LOG_WARNING_S << "getStream() called in parallel mode, returning nullptr";
    return nullptr;
  }
  return m_stream ? m_stream->get() : nullptr;
}

void GpuGenericCudaPreprocessor::synchronize() const {
  if (m_stream) {
    m_stream->synchronize();
  }
}

void GpuGenericCudaPreprocessor::resetCache() const {
  if (m_config.enable_parallel) {
    return; // No cache in parallel mode
  }
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_stream) {
    m_stream->synchronize();
  }
  m_cache.reset();
}

void GpuGenericCudaPreprocessor::validatePreprocessArgs(
    const FramePreprocessArg &args, int src_channels) {
  if (args.model_input_shape.c <= 0 || args.model_input_shape.h <= 0 ||
      args.model_input_shape.w <= 0) {
    LOG_ERROR_S << "Invalid model_input_shape: c=" << args.model_input_shape.c
                << ", h=" << args.model_input_shape.h
                << ", w=" << args.model_input_shape.w;
    throw std::invalid_argument("model_input_shape dimensions must be positive.");
  }

  if (args.mean_vals.empty()) {
    throw std::invalid_argument("mean_vals cannot be empty.");
  }

  if (args.norm_vals.empty()) {
    throw std::invalid_argument("norm_vals (std) cannot be empty.");
  }

  if (args.mean_vals.size() != static_cast<size_t>(args.model_input_shape.c)) {
    throw std::invalid_argument(
        "mean_vals size must match model input channels.");
  }

  if (args.norm_vals.size() != static_cast<size_t>(args.model_input_shape.c)) {
    throw std::invalid_argument(
        "norm_vals size must match model input channels.");
  }

  for (size_t i = 0; i < args.norm_vals.size(); ++i) {
    if (std::abs(args.norm_vals[i]) < 1e-10f) {
      throw std::invalid_argument("norm_vals cannot contain zero values.");
    }
  }

  if (src_channels != args.model_input_shape.c) {
    throw std::invalid_argument(
        "Input image channels must match model input channels.");
  }

  if (args.is_equal_scale) {
    if (args.pad.empty()) {
      throw std::invalid_argument(
          "pad cannot be empty when is_equal_scale is true.");
    }
    if (args.pad.size() != static_cast<size_t>(args.model_input_shape.c)) {
      throw std::invalid_argument("pad size must match model input channels.");
    }
  }

  if (args.data_type != DataType::FLOAT32 &&
      args.data_type != DataType::FLOAT16) {
    throw std::invalid_argument(
        "Unsupported dataType. Only FLOAT32 and FLOAT16 are supported.");
  }
}

// ===========================================================================
// Public Interface - Dispatch to appropriate implementation
// ===========================================================================

TypedBuffer
GpuGenericCudaPreprocessor::process(const FramePreprocessArg &args,
                                    const FrameInput &input,
                                    FrameTransformContext &runtime_args) const {
  if (m_config.enable_parallel) {
    return processParallel(args, input, runtime_args);
  } else {
    return processSequential(args, input, runtime_args);
  }
}

TypedBuffer GpuGenericCudaPreprocessor::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &inputs,
    std::vector<FrameTransformContext> &runtime_args) const {
  if (m_config.enable_parallel) {
    return batchProcessParallel(args, inputs, runtime_args);
  } else {
    return batchProcessSequential(args, inputs, runtime_args);
  }
}

// ===========================================================================
// Sequential Mode - Helper Functions
// ===========================================================================

void GpuGenericCudaPreprocessor::updateParameterBuffers(
    const FramePreprocessArg &args, cudaStream_t stream) const {
  if (m_cache.cached_mean_vals != args.mean_vals) {
    m_cache.d_mean.initFromHost(args.mean_vals, stream);
    m_cache.cached_mean_vals = args.mean_vals;
  }

  if (m_cache.cached_norm_vals != args.norm_vals) {
    m_cache.d_std.initFromHost(args.norm_vals, stream);
    m_cache.cached_norm_vals = args.norm_vals;
  }

  if (args.is_equal_scale && m_cache.cached_pad_vals != args.pad) {
    m_cache.d_pad.initFromHost(args.pad, stream);
    m_cache.cached_pad_vals = args.pad;
  }
}

void GpuGenericCudaPreprocessor::ensureWorkingBufferCapacity(
    const FramePreprocessArg &args, int batch_size, cudaStream_t stream) const {
  size_t single_image_elements = static_cast<size_t>(args.model_input_shape.c) *
                               args.model_input_shape.h * args.model_input_shape.w;
  size_t batch_elements = single_image_elements * batch_size;
  size_t hwc_byte_size_f_p32 = batch_elements * sizeof(float);

  if (m_cache.d_hwc_buffer.capacity() < hwc_byte_size_f_p32) {
    m_cache.d_hwc_buffer.reserve(hwc_byte_size_f_p32, stream);
  }

  if (args.hwc2chw && m_cache.d_chw_buffer.capacity() < hwc_byte_size_f_p32) {
    m_cache.d_chw_buffer.reserve(hwc_byte_size_f_p32, stream);
  }

  if (batch_size > 1) {
    size_t batch_count = static_cast<size_t>(batch_size);
    if (m_cache.d_src_ptrs.capacity() < batch_count)
      m_cache.d_src_ptrs.reserve(batch_count, stream);
    if (m_cache.d_src_heights.capacity() < batch_count)
      m_cache.d_src_heights.reserve(batch_count, stream);
    if (m_cache.d_src_widths.capacity() < batch_count)
      m_cache.d_src_widths.reserve(batch_count, stream);
    if (m_cache.d_rois.capacity() < batch_count)
      m_cache.d_rois.reserve(batch_count, stream);

    if (args.is_equal_scale) {
      if (m_cache.d_new_heights.capacity() < batch_count)
        m_cache.d_new_heights.reserve(batch_count, stream);
      if (m_cache.d_new_widths.capacity() < batch_count)
        m_cache.d_new_widths.reserve(batch_count, stream);
      if (m_cache.d_pad_ys.capacity() < batch_count)
        m_cache.d_pad_ys.reserve(batch_count, stream);
      if (m_cache.d_pad_xs.capacity() < batch_count)
        m_cache.d_pad_xs.reserve(batch_count, stream);
    }
  }
}

// ===========================================================================
// Sequential Mode Implementation
// ===========================================================================

TypedBuffer GpuGenericCudaPreprocessor::processSequential(
    const FramePreprocessArg &args, const FrameInput &input,
    FrameTransformContext &runtime_args) const {
  if (input.image == nullptr) {
    throw std::runtime_error("Input frame is null.");
  }
  validatePreprocessArgs(args, input.image->channels());

  // Lock for thread safety
  std::lock_guard<std::mutex> lock(m_mutex);

  cudaStream_t stream = m_stream->get();

  // Update cached parameters if changed
  updateParameterBuffers(args, stream);
  ensureWorkingBufferCapacity(args, 1, stream);

  // Set ROI
  if (input.input_roi == nullptr) {
    runtime_args.roi =
        std::make_shared<cv::Rect>(0, 0, input.image->cols, input.image->rows);
  } else {
    runtime_args.roi = input.input_roi;
  }
  runtime_args.origin_shape = {input.image->cols, input.image->rows,
                             input.image->channels()};

  const auto &image = *input.image;
  const auto &roi = *runtime_args.roi;

  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    throw std::runtime_error("Invalid ROI.");
  }

  int src_h = roi.height > 0 ? roi.height : image.rows;
  int src_w = roi.width > 0 ? roi.width : image.cols;
  int src_c = image.channels();

  // Upload input image using cached buffer (avoids alloc/free per call)
  size_t input_image_size = image.total() * image.elemSize();
  if (m_cache.d_input_image.capacity() < input_image_size) {
    m_cache.d_input_image.reserve(input_image_size, stream);
  }
  CHECK_CUDA_ERROR(cudaMemcpyAsync(m_cache.d_input_image.unsafePtr(), image.data,
                                   input_image_size, cudaMemcpyHostToDevice,
                                   stream));

  size_t total_elements = static_cast<size_t>(args.model_input_shape.c) *
                         args.model_input_shape.h * args.model_input_shape.w;
  size_t byte_size_f_p32 = total_elements * sizeof(float);
  size_t final_byte_size =
      total_elements * TypedBuffer::getElementSize(args.data_type);

  float *hwc_ptr = reinterpret_cast<float *>(
      m_cache.d_hwc_buffer.writePtr(byte_size_f_p32, stream));

  if (args.is_equal_scale) {
    float scale = std::min(static_cast<float>(args.model_input_shape.w) / src_w,
                           static_cast<float>(args.model_input_shape.h) / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    runtime_args.left_pad = (args.model_input_shape.w - new_w) / 2;
    runtime_args.top_pad = (args.model_input_shape.h - new_h) / 2;

    cuda_op::ROIData roi_data = {roi.x, roi.y, roi.height, roi.width};
    cuda_op::escaleResizeNormalizeGpu(
        static_cast<const uint8_t *>(m_cache.d_input_image.unsafePtr()), hwc_ptr,
        image.cols, src_c, roi_data, args.model_input_shape.h,
        args.model_input_shape.w, m_cache.d_mean.readPtr(),
        m_cache.d_std.readPtr(), m_cache.d_pad.readPtr(), stream);
  } else {
    cuda_op::cropResizeNormalizeGpu(
        static_cast<const uint8_t *>(m_cache.d_input_image.unsafePtr()), hwc_ptr,
        image.rows, image.cols, src_c, roi.x, roi.y, src_h, src_w,
        args.model_input_shape.h, args.model_input_shape.w,
        m_cache.d_mean.readPtr(), m_cache.d_std.readPtr(), stream);
  }

  float *source_ptr = hwc_ptr;
  if (args.hwc2chw) {
    float *chw_ptr = reinterpret_cast<float *>(
        m_cache.d_chw_buffer.writePtr(byte_size_f_p32, stream));
    cuda_op::hwcToChwGpu(hwc_ptr, chw_ptr, args.model_input_shape.h,
                            args.model_input_shape.w, args.model_input_shape.c,
                            stream);
    source_ptr = chw_ptr;
  }

  TypedBuffer final_device_buffer =
      TypedBuffer::createFromGpu(args.data_type, final_byte_size);

  if (args.data_type == DataType::FLOAT16) {
    cuda_op::fp32ToFp16Gpu(
        source_ptr, static_cast<uint16_t *>(final_device_buffer.getRawDevicePtr()),
        total_elements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(final_device_buffer.getRawDevicePtr(),
                                     source_ptr, final_byte_size,
                                     cudaMemcpyDeviceToDevice, stream));
  }

  // Synchronize to ensure all operations complete before mutex is released.
  // This prevents race conditions when another thread acquires the lock and
  // potentially reallocates cached buffers (via updateParameterBuffers).
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.output_location == BufferLocation::GpuDevice) {
    return final_device_buffer;
  } else {
    std::vector<uint8_t> host_data(final_byte_size);
    CHECK_CUDA_ERROR(cudaMemcpy(host_data.data(),
                                final_device_buffer.getRawDevicePtr(),
                                final_byte_size, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.data_type, std::move(host_data));
  }
}

TypedBuffer GpuGenericCudaPreprocessor::batchProcessSequential(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtime_args) const {
  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batch_size = frames.size();

  if (frames[0].image == nullptr) {
    throw std::runtime_error("First input frame is null.");
  }
  validatePreprocessArgs(args, frames[0].image->channels());

  const int expected_channels = frames[0].image->channels();
  for (size_t i = 1; i < batch_size; ++i) {
    if (frames[i].image != nullptr &&
        frames[i].image->channels() != expected_channels) {
      throw std::invalid_argument(
          "All images in batch must have the same number of channels.");
    }
  }

  // Lock for thread safety
  std::lock_guard<std::mutex> lock(m_mutex);

  cudaStream_t stream = m_stream->get();

  updateParameterBuffers(args, stream);
  ensureWorkingBufferCapacity(args, batch_size, stream);

  runtime_args.resize(batch_size);

  // Ensure we have enough cached input image buffers
  if (m_cache.d_batch_input_images.size() < batch_size) {
    m_cache.d_batch_input_images.resize(batch_size);
  }

  std::vector<uint8_t *> h_src_ptrs(batch_size);
  std::vector<int> h_src_heights(batch_size);
  std::vector<int> h_src_widths(batch_size);
  std::vector<cuda_op::ROIData> h_rois(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    const auto &input = frames[i];
    if (input.image == nullptr) {
      throw std::runtime_error("Input frame is null at batch index " +
                               std::to_string(i));
    }

    if (input.input_roi == nullptr) {
      runtime_args[i].roi = std::make_shared<cv::Rect>(0, 0, input.image->cols,
                                                      input.image->rows);
    } else {
      runtime_args[i].roi = input.input_roi;
    }

    const auto &roi = *runtime_args[i].roi;
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        roi.x + roi.width > input.image->cols ||
        roi.y + roi.height > input.image->rows) {
      throw std::runtime_error("Invalid ROI for image at batch index " +
                               std::to_string(i));
    }

    // Use cached buffer, expand if needed
    size_t image_size = input.image->total() * input.image->elemSize();
    if (m_cache.d_batch_input_images[i].capacity() < image_size) {
      m_cache.d_batch_input_images[i].reserve(image_size, stream);
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_cache.d_batch_input_images[i].unsafePtr(),
                                     input.image->data, image_size,
                                     cudaMemcpyHostToDevice, stream));

    h_src_ptrs[i] =
        static_cast<uint8_t *>(m_cache.d_batch_input_images[i].unsafePtr());
    h_src_heights[i] = input.image->rows;
    h_src_widths[i] = input.image->cols;
    h_rois[i] = {roi.x, roi.y, roi.height, roi.width};
    runtime_args[i].origin_shape = {input.image->cols, input.image->rows,
                                  input.image->channels()};
  }

  m_cache.d_src_ptrs.initFromHost(h_src_ptrs, stream);
  m_cache.d_src_heights.initFromHost(h_src_heights, stream);
  m_cache.d_src_widths.initFromHost(h_src_widths, stream);
  m_cache.d_rois.initFromHost(h_rois, stream);

  size_t single_image_elements = static_cast<size_t>(args.model_input_shape.c) *
                               args.model_input_shape.h * args.model_input_shape.w;
  size_t total_elements = single_image_elements * batch_size;
  size_t byte_size_f_p32 = total_elements * sizeof(float);
  size_t final_byte_size =
      total_elements * TypedBuffer::getElementSize(args.data_type);

  float *hwc_batch_ptr = reinterpret_cast<float *>(
      m_cache.d_hwc_buffer.writePtr(byte_size_f_p32, stream));

  if (args.is_equal_scale) {
    std::vector<int> h_new_heights(batch_size), h_new_widths(batch_size);
    std::vector<int> h_pad_ys(batch_size), h_pad_xs(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
      float scale =
          std::min(static_cast<float>(args.model_input_shape.w) / h_rois[i].w,
                   static_cast<float>(args.model_input_shape.h) / h_rois[i].h);
      h_new_widths[i] = static_cast<int>(h_rois[i].w * scale);
      h_new_heights[i] = static_cast<int>(h_rois[i].h * scale);
      h_pad_xs[i] = (args.model_input_shape.w - h_new_widths[i]) / 2;
      h_pad_ys[i] = (args.model_input_shape.h - h_new_heights[i]) / 2;
      runtime_args[i].left_pad = h_pad_xs[i];
      runtime_args[i].top_pad = h_pad_ys[i];
    }

    m_cache.d_new_heights.initFromHost(h_new_heights, stream);
    m_cache.d_new_widths.initFromHost(h_new_widths, stream);
    m_cache.d_pad_ys.initFromHost(h_pad_ys, stream);
    m_cache.d_pad_xs.initFromHost(h_pad_xs, stream);

    cuda_op::batchEscaleResizeNormalizeGpu(
        reinterpret_cast<const uint8_t *const *>(m_cache.d_src_ptrs.readPtr()),
        hwc_batch_ptr, m_cache.d_src_heights.readPtr(),
        m_cache.d_src_widths.readPtr(), args.model_input_shape.c,
        m_cache.d_rois.readPtr(), args.model_input_shape.h,
        args.model_input_shape.w, m_cache.d_mean.readPtr(),
        m_cache.d_std.readPtr(), m_cache.d_pad.readPtr(),
        m_cache.d_new_heights.readPtr(), m_cache.d_new_widths.readPtr(),
        m_cache.d_pad_ys.readPtr(), m_cache.d_pad_xs.readPtr(), batch_size,
        stream);
  } else {
    cuda_op::batchCropResizeNormalizeGpu(
        reinterpret_cast<const uint8_t *const *>(m_cache.d_src_ptrs.readPtr()),
        hwc_batch_ptr, m_cache.d_src_heights.readPtr(),
        m_cache.d_src_widths.readPtr(), args.model_input_shape.c,
        m_cache.d_rois.readPtr(), args.model_input_shape.h,
        args.model_input_shape.w, m_cache.d_mean.readPtr(),
        m_cache.d_std.readPtr(), batch_size, stream);
  }

  float *source_ptr = hwc_batch_ptr;
  if (args.hwc2chw) {
    float *chw_batch_ptr = reinterpret_cast<float *>(
        m_cache.d_chw_buffer.writePtr(byte_size_f_p32, stream));
    cuda_op::batchHwcToChwGpu(
        hwc_batch_ptr, chw_batch_ptr, args.model_input_shape.h,
        args.model_input_shape.w, args.model_input_shape.c, batch_size, stream);
    source_ptr = chw_batch_ptr;
  }

  TypedBuffer final_device_buffer =
      TypedBuffer::createFromGpu(args.data_type, final_byte_size);

  if (args.data_type == DataType::FLOAT16) {
    cuda_op::fp32ToFp16Gpu(
        source_ptr, static_cast<uint16_t *>(final_device_buffer.getRawDevicePtr()),
        total_elements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(final_device_buffer.getRawDevicePtr(),
                                     source_ptr, final_byte_size,
                                     cudaMemcpyDeviceToDevice, stream));
  }

  // Synchronize to ensure all operations complete before mutex is released.
  // This prevents race conditions when another thread acquires the lock and
  // potentially reallocates cached buffers (via updateParameterBuffers).
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.output_location == BufferLocation::GpuDevice) {
    return final_device_buffer;
  } else {
    std::vector<uint8_t> host_data(final_byte_size);
    CHECK_CUDA_ERROR(cudaMemcpy(host_data.data(),
                                final_device_buffer.getRawDevicePtr(),
                                final_byte_size, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.data_type, std::move(host_data));
  }
}

// ===========================================================================
// Parallel Mode Implementation (no caching, each call independent)
// ===========================================================================

TypedBuffer GpuGenericCudaPreprocessor::processParallel(
    const FramePreprocessArg &args, const FrameInput &input,
    FrameTransformContext &runtime_args) const {
  if (input.image == nullptr) {
    throw std::runtime_error("Input frame is null.");
  }
  validatePreprocessArgs(args, input.image->channels());

  // Use default stream (nullptr) for parallel mode
  // Each call is independent, CUDA handles synchronization
  cudaStream_t stream = nullptr;

  // Set ROI
  if (input.input_roi == nullptr) {
    runtime_args.roi =
        std::make_shared<cv::Rect>(0, 0, input.image->cols, input.image->rows);
  } else {
    runtime_args.roi = input.input_roi;
  }
  runtime_args.origin_shape = {input.image->cols, input.image->rows,
                             input.image->channels()};

  const auto &image = *input.image;
  const auto &roi = *runtime_args.roi;

  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    throw std::runtime_error("Invalid ROI.");
  }

  int src_h = roi.height > 0 ? roi.height : image.rows;
  int src_w = roi.width > 0 ? roi.width : image.cols;
  int src_c = image.channels();

  // Allocate all buffers fresh for this call
  cuda_utils::DeviceByteBuffer d_input_image(image.total() * image.elemSize());
  CHECK_CUDA_ERROR(cudaMemcpy(d_input_image.unsafePtr(), image.data,
                              image.total() * image.elemSize(),
                              cudaMemcpyHostToDevice));

  cuda_utils::CudaDeviceBuffer<float> d_mean(args.mean_vals.size());
  cuda_utils::CudaDeviceBuffer<float> d_std(args.norm_vals.size());
  d_mean.initFromHost(args.mean_vals);
  d_std.initFromHost(args.norm_vals);

  size_t total_elements = static_cast<size_t>(args.model_input_shape.c) *
                         args.model_input_shape.h * args.model_input_shape.w;
  size_t byte_size_f_p32 = total_elements * sizeof(float);
  size_t final_byte_size =
      total_elements * TypedBuffer::getElementSize(args.data_type);

  TypedBuffer hwc_buffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byte_size_f_p32);

  if (args.is_equal_scale) {
    float scale = std::min(static_cast<float>(args.model_input_shape.w) / src_w,
                           static_cast<float>(args.model_input_shape.h) / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    runtime_args.left_pad = (args.model_input_shape.w - new_w) / 2;
    runtime_args.top_pad = (args.model_input_shape.h - new_h) / 2;

    cuda_utils::CudaDeviceBuffer<int> d_pad(args.pad.size());
    d_pad.initFromHost(args.pad);

    cuda_op::ROIData roi_data = {roi.x, roi.y, roi.height, roi.width};
    cuda_op::escaleResizeNormalizeGpu(
        static_cast<const uint8_t *>(d_input_image.unsafePtr()),
        static_cast<float *>(hwc_buffer.getRawDevicePtr()), image.cols, src_c,
        roi_data, args.model_input_shape.h, args.model_input_shape.w,
        d_mean.readPtr(), d_std.readPtr(), d_pad.readPtr(), stream);
  } else {
    cuda_op::cropResizeNormalizeGpu(
        static_cast<const uint8_t *>(d_input_image.unsafePtr()),
        static_cast<float *>(hwc_buffer.getRawDevicePtr()), image.rows,
        image.cols, src_c, roi.x, roi.y, src_h, src_w, args.model_input_shape.h,
        args.model_input_shape.w, d_mean.readPtr(), d_std.readPtr(), stream);
  }

  TypedBuffer final_device_buffer =
      TypedBuffer::createFromGpu(args.data_type, final_byte_size);

  TypedBuffer chw_buffer;
  TypedBuffer *source_buffer = &hwc_buffer;

  if (args.hwc2chw) {
    chw_buffer = TypedBuffer::createFromGpu(DataType::FLOAT32, byte_size_f_p32);
    cuda_op::hwcToChwGpu(
        static_cast<const float *>(hwc_buffer.getRawDevicePtr()),
        static_cast<float *>(chw_buffer.getRawDevicePtr()),
        args.model_input_shape.h, args.model_input_shape.w, args.model_input_shape.c,
        stream);
    source_buffer = &chw_buffer;
  }

  if (args.data_type == DataType::FLOAT16) {
    cuda_op::fp32ToFp16Gpu(
        static_cast<const float *>(source_buffer->getRawDevicePtr()),
        static_cast<uint16_t *>(final_device_buffer.getRawDevicePtr()),
        total_elements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpy(final_device_buffer.getRawDevicePtr(),
                                source_buffer->getRawDevicePtr(), final_byte_size,
                                cudaMemcpyDeviceToDevice));
  }

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.output_location == BufferLocation::GpuDevice) {
    return final_device_buffer;
  } else {
    std::vector<uint8_t> host_data(final_byte_size);
    CHECK_CUDA_ERROR(cudaMemcpy(host_data.data(),
                                final_device_buffer.getRawDevicePtr(),
                                final_byte_size, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.data_type, std::move(host_data));
  }
}

TypedBuffer GpuGenericCudaPreprocessor::batchProcessParallel(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtime_args) const {
  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batch_size = frames.size();

  if (frames[0].image == nullptr) {
    throw std::runtime_error("First input frame is null.");
  }
  validatePreprocessArgs(args, frames[0].image->channels());

  const int expected_channels = frames[0].image->channels();
  for (size_t i = 1; i < batch_size; ++i) {
    if (frames[i].image != nullptr &&
        frames[i].image->channels() != expected_channels) {
      throw std::invalid_argument(
          "All images in batch must have the same number of channels.");
    }
  }

  cudaStream_t stream = nullptr;

  runtime_args.resize(batch_size);

  // Allocate all buffers fresh
  std::vector<cuda_utils::DeviceByteBuffer> d_input_images;
  d_input_images.reserve(batch_size);

  std::vector<uint8_t *> h_src_ptrs(batch_size);
  std::vector<int> h_src_heights(batch_size);
  std::vector<int> h_src_widths(batch_size);
  std::vector<cuda_op::ROIData> h_rois(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    const auto &input = frames[i];
    if (input.image == nullptr) {
      throw std::runtime_error("Input frame is null at batch index " +
                               std::to_string(i));
    }

    if (input.input_roi == nullptr) {
      runtime_args[i].roi = std::make_shared<cv::Rect>(0, 0, input.image->cols,
                                                      input.image->rows);
    } else {
      runtime_args[i].roi = input.input_roi;
    }

    const auto &roi = *runtime_args[i].roi;
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        roi.x + roi.width > input.image->cols ||
        roi.y + roi.height > input.image->rows) {
      throw std::runtime_error("Invalid ROI for image at batch index " +
                               std::to_string(i));
    }

    d_input_images.emplace_back(input.image->total() * input.image->elemSize());
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_images.back().unsafePtr(),
                                input.image->data,
                                input.image->total() * input.image->elemSize(),
                                cudaMemcpyHostToDevice));

    h_src_ptrs[i] = static_cast<uint8_t *>(d_input_images.back().unsafePtr());
    h_src_heights[i] = input.image->rows;
    h_src_widths[i] = input.image->cols;
    h_rois[i] = {roi.x, roi.y, roi.height, roi.width};
    runtime_args[i].origin_shape = {input.image->cols, input.image->rows,
                                  input.image->channels()};
  }

  cuda_utils::CudaDeviceBuffer<float> d_mean(args.mean_vals.size());
  cuda_utils::CudaDeviceBuffer<float> d_std(args.norm_vals.size());
  d_mean.initFromHost(args.mean_vals);
  d_std.initFromHost(args.norm_vals);

  cuda_utils::CudaDeviceBuffer<uint8_t *> d_src_ptrs(batch_size);
  cuda_utils::CudaDeviceBuffer<int> d_src_heights(batch_size);
  cuda_utils::CudaDeviceBuffer<int> d_src_widths(batch_size);
  cuda_utils::CudaDeviceBuffer<cuda_op::ROIData> d_rois(batch_size);

  d_src_ptrs.initFromHost(h_src_ptrs);
  d_src_heights.initFromHost(h_src_heights);
  d_src_widths.initFromHost(h_src_widths);
  d_rois.initFromHost(h_rois);

  size_t single_image_elements = static_cast<size_t>(args.model_input_shape.c) *
                               args.model_input_shape.h * args.model_input_shape.w;
  size_t total_elements = single_image_elements * batch_size;
  size_t byte_size_f_p32 = total_elements * sizeof(float);
  size_t final_byte_size =
      total_elements * TypedBuffer::getElementSize(args.data_type);

  TypedBuffer hwc_batch_buffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byte_size_f_p32);

  if (args.is_equal_scale) {
    cuda_utils::CudaDeviceBuffer<int> d_pad(args.pad.size());
    d_pad.initFromHost(args.pad);

    std::vector<int> h_new_heights(batch_size), h_new_widths(batch_size);
    std::vector<int> h_pad_ys(batch_size), h_pad_xs(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
      float scale =
          std::min(static_cast<float>(args.model_input_shape.w) / h_rois[i].w,
                   static_cast<float>(args.model_input_shape.h) / h_rois[i].h);
      h_new_widths[i] = static_cast<int>(h_rois[i].w * scale);
      h_new_heights[i] = static_cast<int>(h_rois[i].h * scale);
      h_pad_xs[i] = (args.model_input_shape.w - h_new_widths[i]) / 2;
      h_pad_ys[i] = (args.model_input_shape.h - h_new_heights[i]) / 2;
      runtime_args[i].left_pad = h_pad_xs[i];
      runtime_args[i].top_pad = h_pad_ys[i];
    }

    cuda_utils::CudaDeviceBuffer<int> d_new_heights(batch_size);
    cuda_utils::CudaDeviceBuffer<int> d_new_widths(batch_size);
    cuda_utils::CudaDeviceBuffer<int> d_pad_ys(batch_size);
    cuda_utils::CudaDeviceBuffer<int> d_pad_xs(batch_size);

    d_new_heights.initFromHost(h_new_heights);
    d_new_widths.initFromHost(h_new_widths);
    d_pad_ys.initFromHost(h_pad_ys);
    d_pad_xs.initFromHost(h_pad_xs);

    cuda_op::batchEscaleResizeNormalizeGpu(
        reinterpret_cast<const uint8_t *const *>(d_src_ptrs.readPtr()),
        static_cast<float *>(hwc_batch_buffer.getRawDevicePtr()),
        d_src_heights.readPtr(), d_src_widths.readPtr(), args.model_input_shape.c,
        d_rois.readPtr(), args.model_input_shape.h, args.model_input_shape.w,
        d_mean.readPtr(), d_std.readPtr(), d_pad.readPtr(),
        d_new_heights.readPtr(), d_new_widths.readPtr(), d_pad_ys.readPtr(),
        d_pad_xs.readPtr(), batch_size, stream);
  } else {
    cuda_op::batchCropResizeNormalizeGpu(
        reinterpret_cast<const uint8_t *const *>(d_src_ptrs.readPtr()),
        static_cast<float *>(hwc_batch_buffer.getRawDevicePtr()),
        d_src_heights.readPtr(), d_src_widths.readPtr(), args.model_input_shape.c,
        d_rois.readPtr(), args.model_input_shape.h, args.model_input_shape.w,
        d_mean.readPtr(), d_std.readPtr(), batch_size, stream);
  }

  TypedBuffer final_device_buffer =
      TypedBuffer::createFromGpu(args.data_type, final_byte_size);

  TypedBuffer chw_batch_buffer;
  TypedBuffer *source_buffer = &hwc_batch_buffer;

  if (args.hwc2chw) {
    chw_batch_buffer =
        TypedBuffer::createFromGpu(DataType::FLOAT32, byte_size_f_p32);
    cuda_op::batchHwcToChwGpu(
        static_cast<const float *>(hwc_batch_buffer.getRawDevicePtr()),
        static_cast<float *>(chw_batch_buffer.getRawDevicePtr()),
        args.model_input_shape.h, args.model_input_shape.w, args.model_input_shape.c,
        batch_size, stream);
    source_buffer = &chw_batch_buffer;
  }

  if (args.data_type == DataType::FLOAT16) {
    cuda_op::fp32ToFp16Gpu(
        static_cast<const float *>(source_buffer->getRawDevicePtr()),
        static_cast<uint16_t *>(final_device_buffer.getRawDevicePtr()),
        total_elements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpy(final_device_buffer.getRawDevicePtr(),
                                source_buffer->getRawDevicePtr(), final_byte_size,
                                cudaMemcpyDeviceToDevice));
  }

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.output_location == BufferLocation::GpuDevice) {
    return final_device_buffer;
  } else {
    std::vector<uint8_t> host_data(final_byte_size);
    CHECK_CUDA_ERROR(cudaMemcpy(host_data.data(),
                                final_device_buffer.getRawDevicePtr(),
                                final_byte_size, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.data_type, std::move(host_data));
  }
}
} // namespace ai_core::dnn::gpu