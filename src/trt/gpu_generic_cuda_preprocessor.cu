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
  cachedMeanVals.clear();
  cachedNormVals.clear();
  cachedPadVals.clear();
  d_hwcBuffer.reset();
  d_chwBuffer.reset();
  d_inputImage.reset();
  d_batchInputImages.clear();
  d_srcPtrs.reset();
  d_srcHeights.reset();
  d_srcWidths.reset();
  d_rois.reset();
  d_newHeights.reset();
  d_newWidths.reset();
  d_padYs.reset();
  d_padXs.reset();
}

// ===========================================================================
// GpuGenericCudaPreprocessor Implementation
// ===========================================================================

GpuGenericCudaPreprocessor::GpuGenericCudaPreprocessor()
    : GpuGenericCudaPreprocessor(Config::defaults()) {}

GpuGenericCudaPreprocessor::GpuGenericCudaPreprocessor(const Config &config)
    : m_config(config) {
  // Only create stream for sequential mode
  if (!config.enableParallel) {
    m_stream = std::make_unique<CudaStream>(
        config.useHighPriorityStream ? CudaStream::Priority::High
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
  if (m_config.enableParallel) {
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
  if (m_config.enableParallel) {
    return; // No cache in parallel mode
  }
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_stream) {
    m_stream->synchronize();
  }
  m_cache.reset();
}

void GpuGenericCudaPreprocessor::validatePreprocessArgs(
    const FramePreprocessArg &args, int srcChannels) {
  if (args.modelInputShape.c <= 0 || args.modelInputShape.h <= 0 ||
      args.modelInputShape.w <= 0) {
    LOG_ERROR_S << "Invalid modelInputShape: c=" << args.modelInputShape.c
                << ", h=" << args.modelInputShape.h
                << ", w=" << args.modelInputShape.w;
    throw std::invalid_argument("modelInputShape dimensions must be positive.");
  }

  if (args.meanVals.empty()) {
    throw std::invalid_argument("meanVals cannot be empty.");
  }

  if (args.normVals.empty()) {
    throw std::invalid_argument("normVals (std) cannot be empty.");
  }

  if (args.meanVals.size() != static_cast<size_t>(args.modelInputShape.c)) {
    throw std::invalid_argument(
        "meanVals size must match model input channels.");
  }

  if (args.normVals.size() != static_cast<size_t>(args.modelInputShape.c)) {
    throw std::invalid_argument(
        "normVals size must match model input channels.");
  }

  for (size_t i = 0; i < args.normVals.size(); ++i) {
    if (std::abs(args.normVals[i]) < 1e-10f) {
      throw std::invalid_argument("normVals cannot contain zero values.");
    }
  }

  if (srcChannels != args.modelInputShape.c) {
    throw std::invalid_argument(
        "Input image channels must match model input channels.");
  }

  if (args.isEqualScale) {
    if (args.pad.empty()) {
      throw std::invalid_argument(
          "pad cannot be empty when isEqualScale is true.");
    }
    if (args.pad.size() != static_cast<size_t>(args.modelInputShape.c)) {
      throw std::invalid_argument("pad size must match model input channels.");
    }
  }

  if (args.dataType != DataType::FLOAT32 &&
      args.dataType != DataType::FLOAT16) {
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
                                    FrameTransformContext &runtimeArgs) const {
  if (m_config.enableParallel) {
    return processParallel(args, input, runtimeArgs);
  } else {
    return processSequential(args, input, runtimeArgs);
  }
}

TypedBuffer GpuGenericCudaPreprocessor::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &inputs,
    std::vector<FrameTransformContext> &runtimeArgs) const {
  if (m_config.enableParallel) {
    return batchProcessParallel(args, inputs, runtimeArgs);
  } else {
    return batchProcessSequential(args, inputs, runtimeArgs);
  }
}

// ===========================================================================
// Sequential Mode - Helper Functions
// ===========================================================================

void GpuGenericCudaPreprocessor::updateParameterBuffers(
    const FramePreprocessArg &args, cudaStream_t stream) const {
  if (m_cache.cachedMeanVals != args.meanVals) {
    m_cache.d_mean.initFromHost(args.meanVals, stream);
    m_cache.cachedMeanVals = args.meanVals;
  }

  if (m_cache.cachedNormVals != args.normVals) {
    m_cache.d_std.initFromHost(args.normVals, stream);
    m_cache.cachedNormVals = args.normVals;
  }

  if (args.isEqualScale && m_cache.cachedPadVals != args.pad) {
    m_cache.d_pad.initFromHost(args.pad, stream);
    m_cache.cachedPadVals = args.pad;
  }
}

void GpuGenericCudaPreprocessor::ensureWorkingBufferCapacity(
    const FramePreprocessArg &args, int batchSize, cudaStream_t stream) const {
  size_t singleImageElements = static_cast<size_t>(args.modelInputShape.c) *
                               args.modelInputShape.h * args.modelInputShape.w;
  size_t batchElements = singleImageElements * batchSize;
  size_t hwcByteSizeFP32 = batchElements * sizeof(float);

  if (m_cache.d_hwcBuffer.capacity() < hwcByteSizeFP32) {
    m_cache.d_hwcBuffer.reserve(hwcByteSizeFP32, stream);
  }

  if (args.hwc2chw && m_cache.d_chwBuffer.capacity() < hwcByteSizeFP32) {
    m_cache.d_chwBuffer.reserve(hwcByteSizeFP32, stream);
  }

  if (batchSize > 1) {
    size_t batchCount = static_cast<size_t>(batchSize);
    if (m_cache.d_srcPtrs.capacity() < batchCount)
      m_cache.d_srcPtrs.reserve(batchCount, stream);
    if (m_cache.d_srcHeights.capacity() < batchCount)
      m_cache.d_srcHeights.reserve(batchCount, stream);
    if (m_cache.d_srcWidths.capacity() < batchCount)
      m_cache.d_srcWidths.reserve(batchCount, stream);
    if (m_cache.d_rois.capacity() < batchCount)
      m_cache.d_rois.reserve(batchCount, stream);

    if (args.isEqualScale) {
      if (m_cache.d_newHeights.capacity() < batchCount)
        m_cache.d_newHeights.reserve(batchCount, stream);
      if (m_cache.d_newWidths.capacity() < batchCount)
        m_cache.d_newWidths.reserve(batchCount, stream);
      if (m_cache.d_padYs.capacity() < batchCount)
        m_cache.d_padYs.reserve(batchCount, stream);
      if (m_cache.d_padXs.capacity() < batchCount)
        m_cache.d_padXs.reserve(batchCount, stream);
    }
  }
}

// ===========================================================================
// Sequential Mode Implementation
// ===========================================================================

TypedBuffer GpuGenericCudaPreprocessor::processSequential(
    const FramePreprocessArg &args, const FrameInput &input,
    FrameTransformContext &runtimeArgs) const {
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
  if (input.inputRoi == nullptr) {
    runtimeArgs.roi =
        std::make_shared<cv::Rect>(0, 0, input.image->cols, input.image->rows);
  } else {
    runtimeArgs.roi = input.inputRoi;
  }
  runtimeArgs.originShape = {input.image->cols, input.image->rows,
                             input.image->channels()};

  const auto &image = *input.image;
  const auto &roi = *runtimeArgs.roi;

  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    throw std::runtime_error("Invalid ROI.");
  }

  int src_h = roi.height > 0 ? roi.height : image.rows;
  int src_w = roi.width > 0 ? roi.width : image.cols;
  int src_c = image.channels();

  // Upload input image using cached buffer (avoids alloc/free per call)
  size_t inputImageSize = image.total() * image.elemSize();
  if (m_cache.d_inputImage.capacity() < inputImageSize) {
    m_cache.d_inputImage.reserve(inputImageSize, stream);
  }
  CHECK_CUDA_ERROR(cudaMemcpyAsync(m_cache.d_inputImage.unsafePtr(), image.data,
                                   inputImageSize, cudaMemcpyHostToDevice,
                                   stream));

  size_t totalElements = static_cast<size_t>(args.modelInputShape.c) *
                         args.modelInputShape.h * args.modelInputShape.w;
  size_t byteSizeFP32 = totalElements * sizeof(float);
  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);

  float *hwcPtr = reinterpret_cast<float *>(
      m_cache.d_hwcBuffer.writePtr(byteSizeFP32, stream));

  if (args.isEqualScale) {
    float scale = std::min(static_cast<float>(args.modelInputShape.w) / src_w,
                           static_cast<float>(args.modelInputShape.h) / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    runtimeArgs.leftPad = (args.modelInputShape.w - new_w) / 2;
    runtimeArgs.topPad = (args.modelInputShape.h - new_h) / 2;

    cuda_op::ROIData roi_data = {roi.x, roi.y, roi.height, roi.width};
    cuda_op::escale_resize_normalize_gpu(
        static_cast<const uint8_t *>(m_cache.d_inputImage.unsafePtr()), hwcPtr,
        image.cols, src_c, roi_data, args.modelInputShape.h,
        args.modelInputShape.w, m_cache.d_mean.readPtr(),
        m_cache.d_std.readPtr(), m_cache.d_pad.readPtr(), stream);
  } else {
    cuda_op::crop_resize_normalize_gpu(
        static_cast<const uint8_t *>(m_cache.d_inputImage.unsafePtr()), hwcPtr,
        image.rows, image.cols, src_c, roi.x, roi.y, src_h, src_w,
        args.modelInputShape.h, args.modelInputShape.w,
        m_cache.d_mean.readPtr(), m_cache.d_std.readPtr(), stream);
  }

  float *sourcePtr = hwcPtr;
  if (args.hwc2chw) {
    float *chwPtr = reinterpret_cast<float *>(
        m_cache.d_chwBuffer.writePtr(byteSizeFP32, stream));
    cuda_op::hwc_to_chw_gpu(hwcPtr, chwPtr, args.modelInputShape.h,
                            args.modelInputShape.w, args.modelInputShape.c,
                            stream);
    sourcePtr = chwPtr;
  }

  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  if (args.dataType == DataType::FLOAT16) {
    cuda_op::fp32_to_fp16_gpu(
        sourcePtr, static_cast<uint16_t *>(finalDeviceBuffer.getRawDevicePtr()),
        totalElements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(finalDeviceBuffer.getRawDevicePtr(),
                                     sourcePtr, finalByteSize,
                                     cudaMemcpyDeviceToDevice, stream));
  }

  // Synchronize to ensure all operations complete before mutex is released.
  // This prevents race conditions when another thread acquires the lock and
  // potentially reallocates cached buffers (via updateParameterBuffers).
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.outputLocation == BufferLocation::GPU_DEVICE) {
    return finalDeviceBuffer;
  } else {
    std::vector<uint8_t> hostData(finalByteSize);
    CHECK_CUDA_ERROR(cudaMemcpy(hostData.data(),
                                finalDeviceBuffer.getRawDevicePtr(),
                                finalByteSize, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
  }
}

TypedBuffer GpuGenericCudaPreprocessor::batchProcessSequential(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtimeArgs) const {
  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batchSize = frames.size();

  if (frames[0].image == nullptr) {
    throw std::runtime_error("First input frame is null.");
  }
  validatePreprocessArgs(args, frames[0].image->channels());

  const int expectedChannels = frames[0].image->channels();
  for (size_t i = 1; i < batchSize; ++i) {
    if (frames[i].image != nullptr &&
        frames[i].image->channels() != expectedChannels) {
      throw std::invalid_argument(
          "All images in batch must have the same number of channels.");
    }
  }

  // Lock for thread safety
  std::lock_guard<std::mutex> lock(m_mutex);

  cudaStream_t stream = m_stream->get();

  updateParameterBuffers(args, stream);
  ensureWorkingBufferCapacity(args, batchSize, stream);

  runtimeArgs.resize(batchSize);

  // Ensure we have enough cached input image buffers
  if (m_cache.d_batchInputImages.size() < batchSize) {
    m_cache.d_batchInputImages.resize(batchSize);
  }

  std::vector<uint8_t *> h_srcPtrs(batchSize);
  std::vector<int> h_srcHeights(batchSize);
  std::vector<int> h_srcWidths(batchSize);
  std::vector<cuda_op::ROIData> h_rois(batchSize);

  for (size_t i = 0; i < batchSize; ++i) {
    const auto &input = frames[i];
    if (input.image == nullptr) {
      throw std::runtime_error("Input frame is null at batch index " +
                               std::to_string(i));
    }

    if (input.inputRoi == nullptr) {
      runtimeArgs[i].roi = std::make_shared<cv::Rect>(0, 0, input.image->cols,
                                                      input.image->rows);
    } else {
      runtimeArgs[i].roi = input.inputRoi;
    }

    const auto &roi = *runtimeArgs[i].roi;
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        roi.x + roi.width > input.image->cols ||
        roi.y + roi.height > input.image->rows) {
      throw std::runtime_error("Invalid ROI for image at batch index " +
                               std::to_string(i));
    }

    // Use cached buffer, expand if needed
    size_t imageSize = input.image->total() * input.image->elemSize();
    if (m_cache.d_batchInputImages[i].capacity() < imageSize) {
      m_cache.d_batchInputImages[i].reserve(imageSize, stream);
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_cache.d_batchInputImages[i].unsafePtr(),
                                     input.image->data, imageSize,
                                     cudaMemcpyHostToDevice, stream));

    h_srcPtrs[i] =
        static_cast<uint8_t *>(m_cache.d_batchInputImages[i].unsafePtr());
    h_srcHeights[i] = input.image->rows;
    h_srcWidths[i] = input.image->cols;
    h_rois[i] = {roi.x, roi.y, roi.height, roi.width};
    runtimeArgs[i].originShape = {input.image->cols, input.image->rows,
                                  input.image->channels()};
  }

  m_cache.d_srcPtrs.initFromHost(h_srcPtrs, stream);
  m_cache.d_srcHeights.initFromHost(h_srcHeights, stream);
  m_cache.d_srcWidths.initFromHost(h_srcWidths, stream);
  m_cache.d_rois.initFromHost(h_rois, stream);

  size_t singleImageElements = static_cast<size_t>(args.modelInputShape.c) *
                               args.modelInputShape.h * args.modelInputShape.w;
  size_t totalElements = singleImageElements * batchSize;
  size_t byteSizeFP32 = totalElements * sizeof(float);
  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);

  float *hwcBatchPtr = reinterpret_cast<float *>(
      m_cache.d_hwcBuffer.writePtr(byteSizeFP32, stream));

  if (args.isEqualScale) {
    std::vector<int> h_newHeights(batchSize), h_newWidths(batchSize);
    std::vector<int> h_padYs(batchSize), h_padXs(batchSize);

    for (size_t i = 0; i < batchSize; ++i) {
      float scale =
          std::min(static_cast<float>(args.modelInputShape.w) / h_rois[i].w,
                   static_cast<float>(args.modelInputShape.h) / h_rois[i].h);
      h_newWidths[i] = static_cast<int>(h_rois[i].w * scale);
      h_newHeights[i] = static_cast<int>(h_rois[i].h * scale);
      h_padXs[i] = (args.modelInputShape.w - h_newWidths[i]) / 2;
      h_padYs[i] = (args.modelInputShape.h - h_newHeights[i]) / 2;
      runtimeArgs[i].leftPad = h_padXs[i];
      runtimeArgs[i].topPad = h_padYs[i];
    }

    m_cache.d_newHeights.initFromHost(h_newHeights, stream);
    m_cache.d_newWidths.initFromHost(h_newWidths, stream);
    m_cache.d_padYs.initFromHost(h_padYs, stream);
    m_cache.d_padXs.initFromHost(h_padXs, stream);

    cuda_op::batch_escale_resize_normalize_gpu(
        reinterpret_cast<const uint8_t *const *>(m_cache.d_srcPtrs.readPtr()),
        hwcBatchPtr, m_cache.d_srcHeights.readPtr(),
        m_cache.d_srcWidths.readPtr(), args.modelInputShape.c,
        m_cache.d_rois.readPtr(), args.modelInputShape.h,
        args.modelInputShape.w, m_cache.d_mean.readPtr(),
        m_cache.d_std.readPtr(), m_cache.d_pad.readPtr(),
        m_cache.d_newHeights.readPtr(), m_cache.d_newWidths.readPtr(),
        m_cache.d_padYs.readPtr(), m_cache.d_padXs.readPtr(), batchSize,
        stream);
  } else {
    cuda_op::batch_crop_resize_normalize_gpu(
        reinterpret_cast<const uint8_t *const *>(m_cache.d_srcPtrs.readPtr()),
        hwcBatchPtr, m_cache.d_srcHeights.readPtr(),
        m_cache.d_srcWidths.readPtr(), args.modelInputShape.c,
        m_cache.d_rois.readPtr(), args.modelInputShape.h,
        args.modelInputShape.w, m_cache.d_mean.readPtr(),
        m_cache.d_std.readPtr(), batchSize, stream);
  }

  float *sourcePtr = hwcBatchPtr;
  if (args.hwc2chw) {
    float *chwBatchPtr = reinterpret_cast<float *>(
        m_cache.d_chwBuffer.writePtr(byteSizeFP32, stream));
    cuda_op::batch_hwc_to_chw_gpu(
        hwcBatchPtr, chwBatchPtr, args.modelInputShape.h,
        args.modelInputShape.w, args.modelInputShape.c, batchSize, stream);
    sourcePtr = chwBatchPtr;
  }

  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  if (args.dataType == DataType::FLOAT16) {
    cuda_op::fp32_to_fp16_gpu(
        sourcePtr, static_cast<uint16_t *>(finalDeviceBuffer.getRawDevicePtr()),
        totalElements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(finalDeviceBuffer.getRawDevicePtr(),
                                     sourcePtr, finalByteSize,
                                     cudaMemcpyDeviceToDevice, stream));
  }

  // Synchronize to ensure all operations complete before mutex is released.
  // This prevents race conditions when another thread acquires the lock and
  // potentially reallocates cached buffers (via updateParameterBuffers).
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.outputLocation == BufferLocation::GPU_DEVICE) {
    return finalDeviceBuffer;
  } else {
    std::vector<uint8_t> hostData(finalByteSize);
    CHECK_CUDA_ERROR(cudaMemcpy(hostData.data(),
                                finalDeviceBuffer.getRawDevicePtr(),
                                finalByteSize, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
  }
}

// ===========================================================================
// Parallel Mode Implementation (no caching, each call independent)
// ===========================================================================

TypedBuffer GpuGenericCudaPreprocessor::processParallel(
    const FramePreprocessArg &args, const FrameInput &input,
    FrameTransformContext &runtimeArgs) const {
  if (input.image == nullptr) {
    throw std::runtime_error("Input frame is null.");
  }
  validatePreprocessArgs(args, input.image->channels());

  // Use default stream (nullptr) for parallel mode
  // Each call is independent, CUDA handles synchronization
  cudaStream_t stream = nullptr;

  // Set ROI
  if (input.inputRoi == nullptr) {
    runtimeArgs.roi =
        std::make_shared<cv::Rect>(0, 0, input.image->cols, input.image->rows);
  } else {
    runtimeArgs.roi = input.inputRoi;
  }
  runtimeArgs.originShape = {input.image->cols, input.image->rows,
                             input.image->channels()};

  const auto &image = *input.image;
  const auto &roi = *runtimeArgs.roi;

  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    throw std::runtime_error("Invalid ROI.");
  }

  int src_h = roi.height > 0 ? roi.height : image.rows;
  int src_w = roi.width > 0 ? roi.width : image.cols;
  int src_c = image.channels();

  // Allocate all buffers fresh for this call
  cuda_utils::DeviceByteBuffer d_inputImage(image.total() * image.elemSize());
  CHECK_CUDA_ERROR(cudaMemcpy(d_inputImage.unsafePtr(), image.data,
                              image.total() * image.elemSize(),
                              cudaMemcpyHostToDevice));

  cuda_utils::CudaDeviceBuffer<float> d_mean(args.meanVals.size());
  cuda_utils::CudaDeviceBuffer<float> d_std(args.normVals.size());
  d_mean.initFromHost(args.meanVals);
  d_std.initFromHost(args.normVals);

  size_t totalElements = static_cast<size_t>(args.modelInputShape.c) *
                         args.modelInputShape.h * args.modelInputShape.w;
  size_t byteSizeFP32 = totalElements * sizeof(float);
  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);

  TypedBuffer hwcBuffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);

  if (args.isEqualScale) {
    float scale = std::min(static_cast<float>(args.modelInputShape.w) / src_w,
                           static_cast<float>(args.modelInputShape.h) / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    runtimeArgs.leftPad = (args.modelInputShape.w - new_w) / 2;
    runtimeArgs.topPad = (args.modelInputShape.h - new_h) / 2;

    cuda_utils::CudaDeviceBuffer<int> d_pad(args.pad.size());
    d_pad.initFromHost(args.pad);

    cuda_op::ROIData roi_data = {roi.x, roi.y, roi.height, roi.width};
    cuda_op::escale_resize_normalize_gpu(
        static_cast<const uint8_t *>(d_inputImage.unsafePtr()),
        static_cast<float *>(hwcBuffer.getRawDevicePtr()), image.cols, src_c,
        roi_data, args.modelInputShape.h, args.modelInputShape.w,
        d_mean.readPtr(), d_std.readPtr(), d_pad.readPtr(), stream);
  } else {
    cuda_op::crop_resize_normalize_gpu(
        static_cast<const uint8_t *>(d_inputImage.unsafePtr()),
        static_cast<float *>(hwcBuffer.getRawDevicePtr()), image.rows,
        image.cols, src_c, roi.x, roi.y, src_h, src_w, args.modelInputShape.h,
        args.modelInputShape.w, d_mean.readPtr(), d_std.readPtr(), stream);
  }

  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  TypedBuffer chwBuffer;
  TypedBuffer *sourceBuffer = &hwcBuffer;

  if (args.hwc2chw) {
    chwBuffer = TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);
    cuda_op::hwc_to_chw_gpu(
        static_cast<const float *>(hwcBuffer.getRawDevicePtr()),
        static_cast<float *>(chwBuffer.getRawDevicePtr()),
        args.modelInputShape.h, args.modelInputShape.w, args.modelInputShape.c,
        stream);
    sourceBuffer = &chwBuffer;
  }

  if (args.dataType == DataType::FLOAT16) {
    cuda_op::fp32_to_fp16_gpu(
        static_cast<const float *>(sourceBuffer->getRawDevicePtr()),
        static_cast<uint16_t *>(finalDeviceBuffer.getRawDevicePtr()),
        totalElements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                                sourceBuffer->getRawDevicePtr(), finalByteSize,
                                cudaMemcpyDeviceToDevice));
  }

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.outputLocation == BufferLocation::GPU_DEVICE) {
    return finalDeviceBuffer;
  } else {
    std::vector<uint8_t> hostData(finalByteSize);
    CHECK_CUDA_ERROR(cudaMemcpy(hostData.data(),
                                finalDeviceBuffer.getRawDevicePtr(),
                                finalByteSize, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
  }
}

TypedBuffer GpuGenericCudaPreprocessor::batchProcessParallel(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtimeArgs) const {
  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batchSize = frames.size();

  if (frames[0].image == nullptr) {
    throw std::runtime_error("First input frame is null.");
  }
  validatePreprocessArgs(args, frames[0].image->channels());

  const int expectedChannels = frames[0].image->channels();
  for (size_t i = 1; i < batchSize; ++i) {
    if (frames[i].image != nullptr &&
        frames[i].image->channels() != expectedChannels) {
      throw std::invalid_argument(
          "All images in batch must have the same number of channels.");
    }
  }

  cudaStream_t stream = nullptr;

  runtimeArgs.resize(batchSize);

  // Allocate all buffers fresh
  std::vector<cuda_utils::DeviceByteBuffer> d_inputImages;
  d_inputImages.reserve(batchSize);

  std::vector<uint8_t *> h_srcPtrs(batchSize);
  std::vector<int> h_srcHeights(batchSize);
  std::vector<int> h_srcWidths(batchSize);
  std::vector<cuda_op::ROIData> h_rois(batchSize);

  for (size_t i = 0; i < batchSize; ++i) {
    const auto &input = frames[i];
    if (input.image == nullptr) {
      throw std::runtime_error("Input frame is null at batch index " +
                               std::to_string(i));
    }

    if (input.inputRoi == nullptr) {
      runtimeArgs[i].roi = std::make_shared<cv::Rect>(0, 0, input.image->cols,
                                                      input.image->rows);
    } else {
      runtimeArgs[i].roi = input.inputRoi;
    }

    const auto &roi = *runtimeArgs[i].roi;
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        roi.x + roi.width > input.image->cols ||
        roi.y + roi.height > input.image->rows) {
      throw std::runtime_error("Invalid ROI for image at batch index " +
                               std::to_string(i));
    }

    d_inputImages.emplace_back(input.image->total() * input.image->elemSize());
    CHECK_CUDA_ERROR(cudaMemcpy(d_inputImages.back().unsafePtr(),
                                input.image->data,
                                input.image->total() * input.image->elemSize(),
                                cudaMemcpyHostToDevice));

    h_srcPtrs[i] = static_cast<uint8_t *>(d_inputImages.back().unsafePtr());
    h_srcHeights[i] = input.image->rows;
    h_srcWidths[i] = input.image->cols;
    h_rois[i] = {roi.x, roi.y, roi.height, roi.width};
    runtimeArgs[i].originShape = {input.image->cols, input.image->rows,
                                  input.image->channels()};
  }

  cuda_utils::CudaDeviceBuffer<float> d_mean(args.meanVals.size());
  cuda_utils::CudaDeviceBuffer<float> d_std(args.normVals.size());
  d_mean.initFromHost(args.meanVals);
  d_std.initFromHost(args.normVals);

  cuda_utils::CudaDeviceBuffer<uint8_t *> d_srcPtrs(batchSize);
  cuda_utils::CudaDeviceBuffer<int> d_srcHeights(batchSize);
  cuda_utils::CudaDeviceBuffer<int> d_srcWidths(batchSize);
  cuda_utils::CudaDeviceBuffer<cuda_op::ROIData> d_rois(batchSize);

  d_srcPtrs.initFromHost(h_srcPtrs);
  d_srcHeights.initFromHost(h_srcHeights);
  d_srcWidths.initFromHost(h_srcWidths);
  d_rois.initFromHost(h_rois);

  size_t singleImageElements = static_cast<size_t>(args.modelInputShape.c) *
                               args.modelInputShape.h * args.modelInputShape.w;
  size_t totalElements = singleImageElements * batchSize;
  size_t byteSizeFP32 = totalElements * sizeof(float);
  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);

  TypedBuffer hwcBatchBuffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);

  if (args.isEqualScale) {
    cuda_utils::CudaDeviceBuffer<int> d_pad(args.pad.size());
    d_pad.initFromHost(args.pad);

    std::vector<int> h_newHeights(batchSize), h_newWidths(batchSize);
    std::vector<int> h_padYs(batchSize), h_padXs(batchSize);

    for (size_t i = 0; i < batchSize; ++i) {
      float scale =
          std::min(static_cast<float>(args.modelInputShape.w) / h_rois[i].w,
                   static_cast<float>(args.modelInputShape.h) / h_rois[i].h);
      h_newWidths[i] = static_cast<int>(h_rois[i].w * scale);
      h_newHeights[i] = static_cast<int>(h_rois[i].h * scale);
      h_padXs[i] = (args.modelInputShape.w - h_newWidths[i]) / 2;
      h_padYs[i] = (args.modelInputShape.h - h_newHeights[i]) / 2;
      runtimeArgs[i].leftPad = h_padXs[i];
      runtimeArgs[i].topPad = h_padYs[i];
    }

    cuda_utils::CudaDeviceBuffer<int> d_newHeights(batchSize);
    cuda_utils::CudaDeviceBuffer<int> d_newWidths(batchSize);
    cuda_utils::CudaDeviceBuffer<int> d_padYs(batchSize);
    cuda_utils::CudaDeviceBuffer<int> d_padXs(batchSize);

    d_newHeights.initFromHost(h_newHeights);
    d_newWidths.initFromHost(h_newWidths);
    d_padYs.initFromHost(h_padYs);
    d_padXs.initFromHost(h_padXs);

    cuda_op::batch_escale_resize_normalize_gpu(
        reinterpret_cast<const uint8_t *const *>(d_srcPtrs.readPtr()),
        static_cast<float *>(hwcBatchBuffer.getRawDevicePtr()),
        d_srcHeights.readPtr(), d_srcWidths.readPtr(), args.modelInputShape.c,
        d_rois.readPtr(), args.modelInputShape.h, args.modelInputShape.w,
        d_mean.readPtr(), d_std.readPtr(), d_pad.readPtr(),
        d_newHeights.readPtr(), d_newWidths.readPtr(), d_padYs.readPtr(),
        d_padXs.readPtr(), batchSize, stream);
  } else {
    cuda_op::batch_crop_resize_normalize_gpu(
        reinterpret_cast<const uint8_t *const *>(d_srcPtrs.readPtr()),
        static_cast<float *>(hwcBatchBuffer.getRawDevicePtr()),
        d_srcHeights.readPtr(), d_srcWidths.readPtr(), args.modelInputShape.c,
        d_rois.readPtr(), args.modelInputShape.h, args.modelInputShape.w,
        d_mean.readPtr(), d_std.readPtr(), batchSize, stream);
  }

  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  TypedBuffer chwBatchBuffer;
  TypedBuffer *sourceBuffer = &hwcBatchBuffer;

  if (args.hwc2chw) {
    chwBatchBuffer =
        TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);
    cuda_op::batch_hwc_to_chw_gpu(
        static_cast<const float *>(hwcBatchBuffer.getRawDevicePtr()),
        static_cast<float *>(chwBatchBuffer.getRawDevicePtr()),
        args.modelInputShape.h, args.modelInputShape.w, args.modelInputShape.c,
        batchSize, stream);
    sourceBuffer = &chwBatchBuffer;
  }

  if (args.dataType == DataType::FLOAT16) {
    cuda_op::fp32_to_fp16_gpu(
        static_cast<const float *>(sourceBuffer->getRawDevicePtr()),
        static_cast<uint16_t *>(finalDeviceBuffer.getRawDevicePtr()),
        totalElements, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                                sourceBuffer->getRawDevicePtr(), finalByteSize,
                                cudaMemcpyDeviceToDevice));
  }

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  if (args.outputLocation == BufferLocation::GPU_DEVICE) {
    return finalDeviceBuffer;
  } else {
    std::vector<uint8_t> hostData(finalByteSize);
    CHECK_CUDA_ERROR(cudaMemcpy(hostData.data(),
                                finalDeviceBuffer.getRawDevicePtr(),
                                finalByteSize, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
  }
}
} // namespace ai_core::dnn::gpu