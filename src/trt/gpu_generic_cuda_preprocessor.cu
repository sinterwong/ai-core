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
#include "cuda_utils.hpp"
#include "gpu_generic_cuda_preprocessor.hpp"
#include "trt_device_buffer.hpp"
#include "trt_utils.hpp"

#include <logger.hpp>
#include <opencv2/core.hpp>
#include <cmath>

namespace ai_core::dnn::gpu
{

  /**
   * @brief 验证预处理参数的有效性
   * @param args 预处理参数
   * @param srcChannels 输入图像的通道数
   * @throws std::invalid_argument 如果参数无效
   */
  static void validatePreprocessArgs(const FramePreprocessArg &args, int srcChannels)
  {
    // 检查模型输入形状
    if (args.modelInputShape.c <= 0 || args.modelInputShape.h <= 0 ||
        args.modelInputShape.w <= 0)
    {
      LOG_ERRORS << "Invalid modelInputShape: c=" << args.modelInputShape.c
                 << ", h=" << args.modelInputShape.h
                 << ", w=" << args.modelInputShape.w;
      throw std::invalid_argument("modelInputShape dimensions must be positive.");
    }

    // 检查 mean 向量
    if (args.meanVals.empty())
    {
      throw std::invalid_argument("meanVals cannot be empty.");
    }

    // 检查 std/norm 向量
    if (args.normVals.empty())
    {
      throw std::invalid_argument("normVals (std) cannot be empty.");
    }

    // 验证 mean 大小与模型通道数匹配
    if (args.meanVals.size() != static_cast<size_t>(args.modelInputShape.c))
    {
      LOG_ERRORS << "meanVals size (" << args.meanVals.size()
                 << ") != modelInputShape.c (" << args.modelInputShape.c << ")";
      throw std::invalid_argument("meanVals size must match model input channels.");
    }

    // 验证 std 大小与模型通道数匹配
    if (args.normVals.size() != static_cast<size_t>(args.modelInputShape.c))
    {
      LOG_ERRORS << "normVals size (" << args.normVals.size()
                 << ") != modelInputShape.c (" << args.modelInputShape.c << ")";
      throw std::invalid_argument("normVals size must match model input channels.");
    }

    // 检查 normVals 中是否有零值（避免除零）
    for (size_t i = 0; i < args.normVals.size(); ++i)
    {
      if (std::abs(args.normVals[i]) < 1e-10f)
      {
        LOG_ERRORS << "normVals[" << i << "] is zero or near-zero, will cause division by zero.";
        throw std::invalid_argument("normVals cannot contain zero values.");
      }
    }

    // 检查输入图像通道数与模型期望是否一致
    if (srcChannels != args.modelInputShape.c)
    {
      LOG_ERRORS << "Input image channels (" << srcChannels
                 << ") != modelInputShape.c (" << args.modelInputShape.c << ")";
      throw std::invalid_argument("Input image channels must match model input channels.");
    }

    // 等比缩放时检查 pad 向量
    if (args.isEqualScale)
    {
      if (args.pad.empty())
      {
        throw std::invalid_argument("pad cannot be empty when isEqualScale is true.");
      }
      if (args.pad.size() != static_cast<size_t>(args.modelInputShape.c))
      {
        LOG_ERRORS << "pad size (" << args.pad.size()
                   << ") != modelInputShape.c (" << args.modelInputShape.c << ")";
        throw std::invalid_argument("pad size must match model input channels.");
      }
    }

    // 检查数据类型是否支持
    if (args.dataType != DataType::FLOAT32 && args.dataType != DataType::FLOAT16)
    {
      throw std::invalid_argument("Unsupported dataType. Only FLOAT32 and FLOAT16 are supported.");
    }
  }

  TypedBuffer GpuGenericCudaPreprocessor::process(const FramePreprocessArg &args,
                                                  const FrameInput &input,
                                                  FrameTransformContext &runtimeArgs) const
  {
    // 检查输入图像是否为空
    if (input.image == nullptr)
    {
      LOG_ERRORS << "Input frame is null.";
      throw std::runtime_error("Input frame is null.");
    }

    // 参数安全性校验
    validatePreprocessArgs(args, input.image->channels());

    // 设置 ROI
    if (input.inputRoi == nullptr)
    {
      runtimeArgs.roi = std::make_shared<cv::Rect>(0, 0, input.image->cols,
                                                   input.image->rows);
    }
    else
    {
      runtimeArgs.roi = input.inputRoi;
    }
    runtimeArgs.originShape = {input.image->cols, input.image->rows,
                               input.image->channels()};

    const auto &image = *input.image;
    const auto &roi = *runtimeArgs.roi;

    // 验证 ROI 有效性
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        roi.x + roi.width > image.cols || roi.y + roi.height > image.rows)
    {
      LOG_ERRORS << "Invalid ROI: " << roi << " for image size: " << image.size();
      throw std::runtime_error("Invalid ROI.");
    }

    const uint8_t *pSrcData = image.data;
    int src_h = image.rows;
    int src_w = image.cols;
    int src_c = image.channels();

    if (roi.area() > 0)
    {
      src_h = roi.height;
      src_w = roi.width;
    }

    // 上传输入图像到 GPU
    trt_utils::TrtDeviceBuffer d_inputImage(image.total() * image.elemSize());
    CHECK_CUDA(cudaMemcpy(d_inputImage.get(), pSrcData,
                          image.total() * image.elemSize(),
                          cudaMemcpyHostToDevice));

    // 上传 mean 和 std 到 GPU
    trt_utils::TrtDeviceBuffer d_mean(args.meanVals.size() * sizeof(float));
    trt_utils::TrtDeviceBuffer d_std(args.normVals.size() * sizeof(float));
    CHECK_CUDA(cudaMemcpy(d_mean.get(), args.meanVals.data(),
                          args.meanVals.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_std.get(), args.normVals.data(),
                          args.normVals.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 分配 HWC 输出缓冲区
    size_t totalElements = (size_t)args.modelInputShape.c *
                           args.modelInputShape.h * args.modelInputShape.w;
    size_t byteSizeFP32 = totalElements * sizeof(float);
    TypedBuffer hwcBuffer =
        TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);

    if (args.isEqualScale)
    {
      // 计算等比缩放参数
      float scale = std::min(static_cast<float>(args.modelInputShape.w) / src_w,
                             static_cast<float>(args.modelInputShape.h) / src_h);
      int new_w = static_cast<int>(src_w * scale);
      int new_h = static_cast<int>(src_h * scale);
      runtimeArgs.leftPad = (args.modelInputShape.w - new_w) / 2;
      runtimeArgs.topPad = (args.modelInputShape.h - new_h) / 2;

      // 上传 pad 到 GPU
      trt_utils::TrtDeviceBuffer d_pad(args.pad.size() * sizeof(float));
      CHECK_CUDA(cudaMemcpy(d_pad.get(), args.pad.data(),
                            args.pad.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

      cuda_op::ROIData roi_data = {roi.x, roi.y, roi.height, roi.width};

      cuda_op::escale_resize_normalize_gpu(
          (const uint8_t *)d_inputImage.get(),
          (float *)hwcBuffer.getRawDevicePtr(),
          image.cols,
          src_c,
          roi_data,
          args.modelInputShape.h,
          args.modelInputShape.w,
          (const float *)d_mean.get(),
          (const float *)d_std.get(),
          (const float *)d_pad.get());
    }
    else
    {
      cuda_op::crop_resize_normalize_gpu(
          (const uint8_t *)d_inputImage.get(),
          (float *)hwcBuffer.getRawDevicePtr(), image.rows, image.cols, src_c,
          roi.x, roi.y, src_h, src_w, args.modelInputShape.h,
          args.modelInputShape.w, (const float *)d_mean.get(),
          (const float *)d_std.get());
    }

    // 分配最终输出缓冲区
    size_t finalByteSize =
        totalElements * TypedBuffer::getElementSize(args.dataType);
    TypedBuffer finalDeviceBuffer =
        TypedBuffer::createFromGpu(args.dataType, finalByteSize);

    if (args.hwc2chw)
    {
      // HWC -> CHW 转换
      TypedBuffer chwBuffer =
          TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);
      cuda_op::hwc_to_chw_gpu((const float *)hwcBuffer.getRawDevicePtr(),
                              (float *)chwBuffer.getRawDevicePtr(),
                              args.modelInputShape.h, args.modelInputShape.w,
                              args.modelInputShape.c);

      if (args.dataType == DataType::FLOAT16)
      {
        cuda_op::fp32_to_fp16_gpu((const float *)chwBuffer.getRawDevicePtr(),
                                  (uint16_t *)finalDeviceBuffer.getRawDevicePtr(),
                                  totalElements);
      }
      else
      {
        CHECK_CUDA(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                              chwBuffer.getRawDevicePtr(), finalByteSize,
                              cudaMemcpyDeviceToDevice));
      }
    }
    else
    {
      if (args.dataType == DataType::FLOAT16)
      {
        cuda_op::fp32_to_fp16_gpu((const float *)hwcBuffer.getRawDevicePtr(),
                                  (uint16_t *)finalDeviceBuffer.getRawDevicePtr(),
                                  totalElements);
      }
      else
      {
        CHECK_CUDA(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                              hwcBuffer.getRawDevicePtr(), finalByteSize,
                              cudaMemcpyDeviceToDevice));
      }
    }

    // 根据输出位置返回结果
    if (args.outputLocation == BufferLocation::GPU_DEVICE)
    {
      return finalDeviceBuffer;
    }
    else
    {
      std::vector<uint8_t> hostData(finalByteSize);
      CHECK_CUDA(cudaMemcpy(hostData.data(), finalDeviceBuffer.getRawDevicePtr(),
                            finalByteSize, cudaMemcpyDeviceToHost));
      return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
    }
  }

  TypedBuffer GpuGenericCudaPreprocessor::batchProcess(
      const FramePreprocessArg &args, const std::vector<FrameInput> &frames,
      std::vector<FrameTransformContext> &runtimeArgs) const
  {
    if (frames.empty())
    {
      return TypedBuffer();
    }

    const size_t batchSize = frames.size();

    // 检查第一张图像有效性并进行参数校验
    if (frames[0].image == nullptr)
    {
      throw std::runtime_error("First input frame is null.");
    }
    validatePreprocessArgs(args, frames[0].image->channels());

    // 验证批次中所有图像通道数一致
    const int expectedChannels = frames[0].image->channels();
    for (size_t i = 1; i < batchSize; ++i)
    {
      if (frames[i].image != nullptr &&
          frames[i].image->channels() != expectedChannels)
      {
        LOG_ERRORS << "Channel mismatch at batch index " << i
                   << ": expected " << expectedChannels
                   << ", got " << frames[i].image->channels();
        throw std::invalid_argument("All images in batch must have the same number of channels.");
      }
    }

    runtimeArgs.resize(batchSize);

    // 准备和上传每帧的数据和元数据
    std::vector<trt_utils::TrtDeviceBuffer> d_input_images;
    d_input_images.reserve(batchSize);

    std::vector<uint8_t *> h_src_ptrs(batchSize);
    std::vector<int> h_src_hs(batchSize);
    std::vector<int> h_src_ws(batchSize);
    std::vector<cuda_op::ROIData> h_rois(batchSize);

    for (size_t i = 0; i < batchSize; ++i)
    {
      const auto &input = frames[i];
      if (input.image == nullptr)
      {
        throw std::runtime_error("Input frame is null at batch index " + std::to_string(i));
      }

      // 设置和验证ROI
      if (input.inputRoi == nullptr)
      {
        runtimeArgs[i].roi = std::make_shared<cv::Rect>(0, 0, input.image->cols, input.image->rows);
      }
      else
      {
        runtimeArgs[i].roi = input.inputRoi;
      }
      const auto &roi = *runtimeArgs[i].roi;
      if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
          roi.x + roi.width > input.image->cols || roi.y + roi.height > input.image->rows)
      {
        throw std::runtime_error("Invalid ROI for image at batch index " + std::to_string(i));
      }

      // 上传单张图片到GPU
      d_input_images.emplace_back(input.image->total() * input.image->elemSize());
      CHECK_CUDA(cudaMemcpy(d_input_images.back().get(), input.image->data,
                            input.image->total() * input.image->elemSize(),
                            cudaMemcpyHostToDevice));

      // 在主机端收集元数据
      h_src_ptrs[i] = static_cast<uint8_t *>(d_input_images.back().get());
      h_src_hs[i] = input.image->rows;
      h_src_ws[i] = input.image->cols;
      h_rois[i] = {roi.x, roi.y, roi.height, roi.width};
      runtimeArgs[i].originShape = {input.image->cols, input.image->rows, input.image->channels()};
    }

    // 上传公共参数和元数据数组
    trt_utils::TrtDeviceBuffer d_mean(args.meanVals.size() * sizeof(float));
    trt_utils::TrtDeviceBuffer d_std(args.normVals.size() * sizeof(float));
    CHECK_CUDA(cudaMemcpy(d_mean.get(), args.meanVals.data(), d_mean.getSizeBytes(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_std.get(), args.normVals.data(), d_std.getSizeBytes(), cudaMemcpyHostToDevice));

    trt_utils::TrtDeviceBuffer d_src_ptrs(h_src_ptrs.size() * sizeof(uint8_t *));
    trt_utils::TrtDeviceBuffer d_src_hs(h_src_hs.size() * sizeof(int));
    trt_utils::TrtDeviceBuffer d_src_ws(h_src_ws.size() * sizeof(int));
    trt_utils::TrtDeviceBuffer d_rois(h_rois.size() * sizeof(cuda_op::ROIData));

    CHECK_CUDA(cudaMemcpy(d_src_ptrs.get(), h_src_ptrs.data(), d_src_ptrs.getSizeBytes(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_src_hs.get(), h_src_hs.data(), d_src_hs.getSizeBytes(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_src_ws.get(), h_src_ws.data(), d_src_ws.getSizeBytes(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rois.get(), h_rois.data(), d_rois.getSizeBytes(), cudaMemcpyHostToDevice));

    // 分配批处理输出缓冲区并调用核心处理Kernel
    size_t singleImageElements = (size_t)args.modelInputShape.c * args.modelInputShape.h * args.modelInputShape.w;
    size_t totalElements = singleImageElements * batchSize;
    size_t byteSizeFP32 = totalElements * sizeof(float);
    TypedBuffer hwcBatchBuffer = TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);

    if (args.isEqualScale)
    {
      trt_utils::TrtDeviceBuffer d_pad(args.pad.size() * sizeof(float));
      CHECK_CUDA(cudaMemcpy(d_pad.get(), args.pad.data(), d_pad.getSizeBytes(), cudaMemcpyHostToDevice));

      // 计算每张图的缩放后尺寸和padding
      std::vector<int> h_new_hs(batchSize), h_new_ws(batchSize), h_pad_ys(batchSize), h_pad_xs(batchSize);
      for (size_t i = 0; i < batchSize; ++i)
      {
        float scale = std::min(static_cast<float>(args.modelInputShape.w) / h_rois[i].w,
                               static_cast<float>(args.modelInputShape.h) / h_rois[i].h);
        h_new_ws[i] = static_cast<int>(h_rois[i].w * scale);
        h_new_hs[i] = static_cast<int>(h_rois[i].h * scale);
        h_pad_xs[i] = (args.modelInputShape.w - h_new_ws[i]) / 2;
        h_pad_ys[i] = (args.modelInputShape.h - h_new_hs[i]) / 2;
        runtimeArgs[i].leftPad = h_pad_xs[i];
        runtimeArgs[i].topPad = h_pad_ys[i];
      }

      trt_utils::TrtDeviceBuffer d_new_hs(h_new_hs.size() * sizeof(int)), d_new_ws(h_new_ws.size() * sizeof(int));
      trt_utils::TrtDeviceBuffer d_pad_ys(h_pad_ys.size() * sizeof(int)), d_pad_xs(h_pad_xs.size() * sizeof(int));
      CHECK_CUDA(cudaMemcpy(d_new_hs.get(), h_new_hs.data(), d_new_hs.getSizeBytes(), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_new_ws.get(), h_new_ws.data(), d_new_ws.getSizeBytes(), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_pad_ys.get(), h_pad_ys.data(), d_pad_ys.getSizeBytes(), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_pad_xs.get(), h_pad_xs.data(), d_pad_xs.getSizeBytes(), cudaMemcpyHostToDevice));

      cuda_op::batch_escale_resize_normalize_gpu(
          (const uint8_t *const *)d_src_ptrs.get(), (float *)hwcBatchBuffer.getRawDevicePtr(),
          (const int *)d_src_hs.get(), (const int *)d_src_ws.get(), args.modelInputShape.c,
          (const cuda_op::ROIData *)d_rois.get(),
          args.modelInputShape.h, args.modelInputShape.w,
          (const float *)d_mean.get(), (const float *)d_std.get(), (const float *)d_pad.get(),
          (const int *)d_new_hs.get(), (const int *)d_new_ws.get(),
          (const int *)d_pad_ys.get(), (const int *)d_pad_xs.get(),
          batchSize);
    }
    else
    {
      cuda_op::batch_crop_resize_normalize_gpu(
          (const uint8_t *const *)d_src_ptrs.get(), (float *)hwcBatchBuffer.getRawDevicePtr(),
          (const int *)d_src_hs.get(), (const int *)d_src_ws.get(), args.modelInputShape.c,
          (const cuda_op::ROIData *)d_rois.get(),
          args.modelInputShape.h, args.modelInputShape.w,
          (const float *)d_mean.get(), (const float *)d_std.get(), batchSize);
    }

    // 处理布局和类型转换
    size_t finalByteSize = totalElements * TypedBuffer::getElementSize(args.dataType);
    TypedBuffer finalDeviceBuffer = TypedBuffer::createFromGpu(args.dataType, finalByteSize);

    TypedBuffer chwBatchBuffer;
    TypedBuffer *sourceBufferForConversion = &hwcBatchBuffer;

    if (args.hwc2chw)
    {
      chwBatchBuffer = TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);
      cuda_op::batch_hwc_to_chw_gpu((const float *)hwcBatchBuffer.getRawDevicePtr(),
                                    (float *)chwBatchBuffer.getRawDevicePtr(),
                                    args.modelInputShape.h, args.modelInputShape.w,
                                    args.modelInputShape.c, batchSize);
      sourceBufferForConversion = &chwBatchBuffer;
    }

    if (args.dataType == DataType::FLOAT16)
    {
      cuda_op::fp32_to_fp16_gpu((const float *)sourceBufferForConversion->getRawDevicePtr(),
                                (uint16_t *)finalDeviceBuffer.getRawDevicePtr(),
                                totalElements);
    }
    else
    { // FLOAT32
      if (sourceBufferForConversion != &finalDeviceBuffer)
      {
        CHECK_CUDA(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                              sourceBufferForConversion->getRawDevicePtr(),
                              finalByteSize, cudaMemcpyDeviceToDevice));
      }
    }

    if (args.outputLocation == BufferLocation::GPU_DEVICE)
    {
      return finalDeviceBuffer;
    }
    else
    {
      std::vector<uint8_t> hostData(finalByteSize);
      CHECK_CUDA(cudaMemcpy(hostData.data(), finalDeviceBuffer.getRawDevicePtr(),
                            finalByteSize, cudaMemcpyDeviceToHost));
      return TypedBuffer::createFromCpu(args.dataType, std::move(hostData));
    }
  }

} // namespace ai_core::dnn::gpu
