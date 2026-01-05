/**
 * @file gpu_generic_cuda_preprocessor.cu
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/typed_buffer.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_utils.hpp"
#include "gpu_generic_cuda_preprocessor.hpp"
#include "trt_utils.hpp"

#include "ai_core/logger.hpp"
#include <cmath>
#include <opencv2/core.hpp>

namespace ai_core::dnn::gpu {
// 类型别名，简化代码
using DeviceByteBuffer = cuda_utils::DeviceByteBuffer;

/**
 * @brief 验证预处理参数的有效性
 * @param args 预处理参数
 * @param srcChannels 输入图像的通道数
 * @throws std::invalid_argument 如果参数无效
 */
static void validatePreprocessArgs(const FramePreprocessArg &args,
                                   int srcChannels) {
  // 检查模型输入形状
  if (args.modelInputShape.c <= 0 || args.modelInputShape.h <= 0 ||
      args.modelInputShape.w <= 0) {
    LOG_ERROR_S << "Invalid modelInputShape: c=" << args.modelInputShape.c
                << ", h=" << args.modelInputShape.h
                << ", w=" << args.modelInputShape.w;
    throw std::invalid_argument("modelInputShape dimensions must be positive.");
  }

  // 检查 mean 向量
  if (args.meanVals.empty()) {
    throw std::invalid_argument("meanVals cannot be empty.");
  }

  // 检查 std/norm 向量
  if (args.normVals.empty()) {
    throw std::invalid_argument("normVals (std) cannot be empty.");
  }

  // 验证 mean 大小与模型通道数匹配
  if (args.meanVals.size() != static_cast<size_t>(args.modelInputShape.c)) {
    LOG_ERROR_S << "meanVals size (" << args.meanVals.size()
                << ") != modelInputShape.c (" << args.modelInputShape.c << ")";
    throw std::invalid_argument(
        "meanVals size must match model input channels.");
  }

  // 验证 std 大小与模型通道数匹配
  if (args.normVals.size() != static_cast<size_t>(args.modelInputShape.c)) {
    LOG_ERROR_S << "normVals size (" << args.normVals.size()
                << ") != modelInputShape.c (" << args.modelInputShape.c << ")";
    throw std::invalid_argument(
        "normVals size must match model input channels.");
  }

  // 检查 normVals 中是否有零值（避免除零）
  for (size_t i = 0; i < args.normVals.size(); ++i) {
    if (std::abs(args.normVals[i]) < 1e-10f) {
      LOG_ERROR_S << "normVals[" << i
                  << "] is zero or near-zero, will cause division by zero.";
      throw std::invalid_argument("normVals cannot contain zero values.");
    }
  }

  // 检查输入图像通道数与模型期望是否一致
  if (srcChannels != args.modelInputShape.c) {
    LOG_ERROR_S << "Input image channels (" << srcChannels
                << ") != modelInputShape.c (" << args.modelInputShape.c << ")";
    throw std::invalid_argument(
        "Input image channels must match model input channels.");
  }

  // 等比缩放时检查 pad 向量
  if (args.isEqualScale) {
    if (args.pad.empty()) {
      throw std::invalid_argument(
          "pad cannot be empty when isEqualScale is true.");
    }
    if (args.pad.size() != static_cast<size_t>(args.modelInputShape.c)) {
      LOG_ERROR_S << "pad size (" << args.pad.size()
                  << ") != modelInputShape.c (" << args.modelInputShape.c
                  << ")";
      throw std::invalid_argument("pad size must match model input channels.");
    }
  }

  // 检查数据类型是否支持
  if (args.dataType != DataType::FLOAT32 &&
      args.dataType != DataType::FLOAT16) {
    throw std::invalid_argument(
        "Unsupported dataType. Only FLOAT32 and FLOAT16 are supported.");
  }
}

TypedBuffer
GpuGenericCudaPreprocessor::process(const FramePreprocessArg &args,
                                    const FrameInput &input,
                                    FrameTransformContext &runtimeArgs) const {
  // 检查输入图像是否为空
  if (input.image == nullptr) {
    LOG_ERROR_S << "Input frame is null.";
    throw std::runtime_error("Input frame is null.");
  }

  // 参数安全性校验
  validatePreprocessArgs(args, input.image->channels());

  // 设置 ROI
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

  // 验证 ROI 有效性
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    LOG_ERROR_S << "Invalid ROI: " << roi
                << " for image size: " << image.size();
    throw std::runtime_error("Invalid ROI.");
  }

  const uint8_t *pSrcData = image.data;
  int src_h = image.rows;
  int src_w = image.cols;
  int src_c = image.channels();

  if (roi.area() > 0) {
    src_h = roi.height;
    src_w = roi.width;
  }

  // 上传输入图像到 GPU - 使用 CudaDeviceBuffer
  DeviceByteBuffer d_inputImage(image.total() * image.elemSize());
  CHECK_CUDA(cudaMemcpy(d_inputImage.unsafePtr(), pSrcData,
                        image.total() * image.elemSize(),
                        cudaMemcpyHostToDevice));

  // 上传 mean 和 std 到 GPU - 使用类型安全的 CudaDeviceBuffer<float>
  cuda_utils::CudaDeviceBuffer<float> d_mean(args.meanVals.size());
  cuda_utils::CudaDeviceBuffer<float> d_std(args.normVals.size());
  d_mean.initFromHost(args.meanVals);
  d_std.initFromHost(args.normVals);

  // 分配 HWC 输出缓冲区
  size_t totalElements = (size_t)args.modelInputShape.c *
                         args.modelInputShape.h * args.modelInputShape.w;
  size_t byteSizeFP32 = totalElements * sizeof(float);
  TypedBuffer hwcBuffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);

  if (args.isEqualScale) {
    // 计算等比缩放参数
    float scale = std::min(static_cast<float>(args.modelInputShape.w) / src_w,
                           static_cast<float>(args.modelInputShape.h) / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    runtimeArgs.leftPad = (args.modelInputShape.w - new_w) / 2;
    runtimeArgs.topPad = (args.modelInputShape.h - new_h) / 2;

    // 上传 pad 到 GPU
    cuda_utils::CudaDeviceBuffer<int> d_pad(args.pad.size());
    d_pad.initFromHost(args.pad);

    cuda_op::ROIData roi_data = {roi.x, roi.y, roi.height, roi.width};

    cuda_op::escale_resize_normalize_gpu(
        (const uint8_t *)d_inputImage.unsafePtr(),
        (float *)hwcBuffer.getRawDevicePtr(), image.cols, src_c, roi_data,
        args.modelInputShape.h, args.modelInputShape.w,
        (const float *)d_mean.readPtr(), (const float *)d_std.readPtr(),
        (const int *)d_pad.readPtr());
  } else {
    cuda_op::crop_resize_normalize_gpu(
        (const uint8_t *)d_inputImage.unsafePtr(),
        (float *)hwcBuffer.getRawDevicePtr(), image.rows, image.cols, src_c,
        roi.x, roi.y, src_h, src_w, args.modelInputShape.h,
        args.modelInputShape.w, (const float *)d_mean.readPtr(),
        (const float *)d_std.readPtr());
  }

  // 分配最终输出缓冲区
  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);
  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  if (args.hwc2chw) {
    // HWC -> CHW 转换
    TypedBuffer chwBuffer =
        TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);
    cuda_op::hwc_to_chw_gpu((const float *)hwcBuffer.getRawDevicePtr(),
                            (float *)chwBuffer.getRawDevicePtr(),
                            args.modelInputShape.h, args.modelInputShape.w,
                            args.modelInputShape.c);

    if (args.dataType == DataType::FLOAT16) {
      cuda_op::fp32_to_fp16_gpu((const float *)chwBuffer.getRawDevicePtr(),
                                (uint16_t *)finalDeviceBuffer.getRawDevicePtr(),
                                totalElements);
    } else {
      CHECK_CUDA(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                            chwBuffer.getRawDevicePtr(), finalByteSize,
                            cudaMemcpyDeviceToDevice));
    }
  } else {
    if (args.dataType == DataType::FLOAT16) {
      cuda_op::fp32_to_fp16_gpu((const float *)hwcBuffer.getRawDevicePtr(),
                                (uint16_t *)finalDeviceBuffer.getRawDevicePtr(),
                                totalElements);
    } else {
      CHECK_CUDA(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                            hwcBuffer.getRawDevicePtr(), finalByteSize,
                            cudaMemcpyDeviceToDevice));
    }
  }

  // 根据输出位置返回结果
  if (args.outputLocation == BufferLocation::GPU_DEVICE) {
    return finalDeviceBuffer;
  } else {
    std::vector<uint8_t> hostData(finalByteSize);
    CHECK_CUDA(cudaMemcpy(hostData.data(), finalDeviceBuffer.getRawDevicePtr(),
                          finalByteSize, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
  }
}

TypedBuffer GpuGenericCudaPreprocessor::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
    std::vector<FrameTransformContext> &runtimeArgs) const {
  if (frames.empty()) {
    return TypedBuffer();
  }

  const size_t batchSize = frames.size();

  // 检查第一张图像有效性并进行参数校验
  if (frames[0].image == nullptr) {
    throw std::runtime_error("First input frame is null.");
  }
  validatePreprocessArgs(args, frames[0].image->channels());

  // 验证批次中所有图像通道数一致
  const int expectedChannels = frames[0].image->channels();
  for (size_t i = 1; i < batchSize; ++i) {
    if (frames[i].image != nullptr &&
        frames[i].image->channels() != expectedChannels) {
      LOG_ERROR_S << "Channel mismatch at batch index " << i << ": expected "
                  << expectedChannels << ", got "
                  << frames[i].image->channels();
      throw std::invalid_argument(
          "All images in batch must have the same number of channels.");
    }
  }

  runtimeArgs.resize(batchSize);

  // 准备和上传每帧的数据和元数据
  std::vector<DeviceByteBuffer> d_input_images;
  d_input_images.reserve(batchSize);

  std::vector<uint8_t *> h_src_ptrs(batchSize);
  std::vector<int> h_src_hs(batchSize);
  std::vector<int> h_src_ws(batchSize);
  std::vector<cuda_op::ROIData> h_rois(batchSize);

  for (size_t i = 0; i < batchSize; ++i) {
    const auto &input = frames[i];
    if (input.image == nullptr) {
      throw std::runtime_error("Input frame is null at batch index " +
                               std::to_string(i));
    }

    // 设置和验证ROI
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

    // 上传单张图片到GPU
    d_input_images.emplace_back(input.image->total() * input.image->elemSize());
    CHECK_CUDA(cudaMemcpy(d_input_images.back().unsafePtr(), input.image->data,
                          input.image->total() * input.image->elemSize(),
                          cudaMemcpyHostToDevice));

    // 在主机端收集元数据
    h_src_ptrs[i] = static_cast<uint8_t *>(d_input_images.back().unsafePtr());
    h_src_hs[i] = input.image->rows;
    h_src_ws[i] = input.image->cols;
    h_rois[i] = {roi.x, roi.y, roi.height, roi.width};
    runtimeArgs[i].originShape = {input.image->cols, input.image->rows,
                                  input.image->channels()};
  }

  // 上传公共参数和元数据数组 - 使用类型安全的 CudaDeviceBuffer
  cuda_utils::CudaDeviceBuffer<float> d_mean(args.meanVals.size());
  cuda_utils::CudaDeviceBuffer<float> d_std(args.normVals.size());
  d_mean.initFromHost(args.meanVals);
  d_std.initFromHost(args.normVals);

  cuda_utils::CudaDeviceBuffer<uint8_t *> d_src_ptrs(h_src_ptrs.size());
  cuda_utils::CudaDeviceBuffer<int> d_src_hs(h_src_hs.size());
  cuda_utils::CudaDeviceBuffer<int> d_src_ws(h_src_ws.size());
  cuda_utils::CudaDeviceBuffer<cuda_op::ROIData> d_rois(h_rois.size());

  d_src_ptrs.initFromHost(h_src_ptrs);
  d_src_hs.initFromHost(h_src_hs);
  d_src_ws.initFromHost(h_src_ws);
  d_rois.initFromHost(h_rois);

  // 分配批处理输出缓冲区并调用核心处理Kernel
  size_t singleImageElements = (size_t)args.modelInputShape.c *
                               args.modelInputShape.h * args.modelInputShape.w;
  size_t totalElements = singleImageElements * batchSize;
  size_t byteSizeFP32 = totalElements * sizeof(float);
  TypedBuffer hwcBatchBuffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);

  if (args.isEqualScale) {
    cuda_utils::CudaDeviceBuffer<int> d_pad(args.pad.size());
    d_pad.initFromHost(args.pad);

    // 计算每张图的缩放后尺寸和padding
    std::vector<int> h_new_hs(batchSize), h_new_ws(batchSize),
        h_pad_ys(batchSize), h_pad_xs(batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
      float scale =
          std::min(static_cast<float>(args.modelInputShape.w) / h_rois[i].w,
                   static_cast<float>(args.modelInputShape.h) / h_rois[i].h);
      h_new_ws[i] = static_cast<int>(h_rois[i].w * scale);
      h_new_hs[i] = static_cast<int>(h_rois[i].h * scale);
      h_pad_xs[i] = (args.modelInputShape.w - h_new_ws[i]) / 2;
      h_pad_ys[i] = (args.modelInputShape.h - h_new_hs[i]) / 2;
      runtimeArgs[i].leftPad = h_pad_xs[i];
      runtimeArgs[i].topPad = h_pad_ys[i];
    }

    cuda_utils::CudaDeviceBuffer<int> d_new_hs(h_new_hs.size());
    cuda_utils::CudaDeviceBuffer<int> d_new_ws(h_new_ws.size());
    cuda_utils::CudaDeviceBuffer<int> d_pad_ys(h_pad_ys.size());
    cuda_utils::CudaDeviceBuffer<int> d_pad_xs(h_pad_xs.size());
    d_new_hs.initFromHost(h_new_hs);
    d_new_ws.initFromHost(h_new_ws);
    d_pad_ys.initFromHost(h_pad_ys);
    d_pad_xs.initFromHost(h_pad_xs);

    cuda_op::batch_escale_resize_normalize_gpu(
        (const uint8_t *const *)d_src_ptrs.readPtr(),
        (float *)hwcBatchBuffer.getRawDevicePtr(),
        (const int *)d_src_hs.readPtr(), (const int *)d_src_ws.readPtr(),
        args.modelInputShape.c, (const cuda_op::ROIData *)d_rois.readPtr(),
        args.modelInputShape.h, args.modelInputShape.w,
        (const float *)d_mean.readPtr(), (const float *)d_std.readPtr(),
        (const int *)d_pad.readPtr(), (const int *)d_new_hs.readPtr(),
        (const int *)d_new_ws.readPtr(), (const int *)d_pad_ys.readPtr(),
        (const int *)d_pad_xs.readPtr(), batchSize);
  } else {
    cuda_op::batch_crop_resize_normalize_gpu(
        (const uint8_t *const *)d_src_ptrs.readPtr(),
        (float *)hwcBatchBuffer.getRawDevicePtr(),
        (const int *)d_src_hs.readPtr(), (const int *)d_src_ws.readPtr(),
        args.modelInputShape.c, (const cuda_op::ROIData *)d_rois.readPtr(),
        args.modelInputShape.h, args.modelInputShape.w,
        (const float *)d_mean.readPtr(), (const float *)d_std.readPtr(),
        batchSize);
  }

  // 处理布局和类型转换
  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);
  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  TypedBuffer chwBatchBuffer;
  TypedBuffer *sourceBufferForConversion = &hwcBatchBuffer;

  if (args.hwc2chw) {
    chwBatchBuffer =
        TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);
    cuda_op::batch_hwc_to_chw_gpu(
        (const float *)hwcBatchBuffer.getRawDevicePtr(),
        (float *)chwBatchBuffer.getRawDevicePtr(), args.modelInputShape.h,
        args.modelInputShape.w, args.modelInputShape.c, batchSize);
    sourceBufferForConversion = &chwBatchBuffer;
  }

  if (args.dataType == DataType::FLOAT16) {
    cuda_op::fp32_to_fp16_gpu(
        (const float *)sourceBufferForConversion->getRawDevicePtr(),
        (uint16_t *)finalDeviceBuffer.getRawDevicePtr(), totalElements);
  } else { // FLOAT32
    if (sourceBufferForConversion != &finalDeviceBuffer) {
      CHECK_CUDA(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                            sourceBufferForConversion->getRawDevicePtr(),
                            finalByteSize, cudaMemcpyDeviceToDevice));
    }
  }

  if (args.outputLocation == BufferLocation::GPU_DEVICE) {
    return finalDeviceBuffer;
  } else {
    std::vector<uint8_t> hostData(finalByteSize);
    CHECK_CUDA(cudaMemcpy(hostData.data(), finalDeviceBuffer.getRawDevicePtr(),
                          finalByteSize, cudaMemcpyDeviceToHost));
    return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
  }
}

} // namespace ai_core::dnn::gpu