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

namespace ai_core::dnn::gpu {

TypedBuffer GpuGenericCudaPreprocessor::process(FramePreprocessArg &args,
                                                const FrameInput &input) const {
  const auto &image = *input.image;
  const auto &roi = *args.roi;

  const uint8_t *pSrcData = image.data;
  int src_h = image.rows;
  int src_w = image.cols;
  int src_c = image.channels();

  if (roi.area() > 0) {
    src_h = roi.height;
    src_w = roi.width;
  }

  trt_utils::TrtDeviceBuffer d_inputImage(image.total() * image.elemSize());
  CHECK_CUDA(cudaMemcpy(d_inputImage.get(), pSrcData,
                        image.total() * image.elemSize(),
                        cudaMemcpyHostToDevice));

  trt_utils::TrtDeviceBuffer d_mean(args.meanVals.size() * sizeof(float));
  trt_utils::TrtDeviceBuffer d_std(args.normVals.size() * sizeof(float));
  CHECK_CUDA(cudaMemcpy(d_mean.get(), args.meanVals.data(),
                        args.meanVals.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_std.get(), args.normVals.data(),
                        args.normVals.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  size_t totalElements = (size_t)args.modelInputShape.c *
                         args.modelInputShape.h * args.modelInputShape.w;
  size_t byteSizeFP32 = totalElements * sizeof(float);
  TypedBuffer hwcBuffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byteSizeFP32);

  if (args.isEqualScale) {
    float scale = std::min(static_cast<float>(args.modelInputShape.w) / src_w,
                           static_cast<float>(args.modelInputShape.h) / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    args.leftPad = (args.modelInputShape.w - new_w) / 2;
    args.topPad = (args.modelInputShape.h - new_h) / 2;

    trt_utils::TrtDeviceBuffer d_pad(args.pad.size() * sizeof(float));
    CHECK_CUDA(cudaMemcpy(d_pad.get(), args.pad.data(),
                          args.pad.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const uint8_t *d_roiImage_ptr =
        (const uint8_t *)d_inputImage.get() +
        ((size_t)roi.y * image.cols + roi.x) * src_c;

    // 5. 调用融合了等比缩放、双线性插值、padding和归一化的kernel
    // 注意：传入的是ROI的尺寸(src_h, src_w)和ROI的起始地址
    cuda_op::escale_resize_normalize_gpu(
        d_roiImage_ptr, (float *)hwcBuffer.getRawDevicePtr(), src_h, src_w,
        src_c,                                          // 源(ROI)尺寸
        args.modelInputShape.h, args.modelInputShape.w, // 目标尺寸
        (const float *)d_mean.get(), (const float *)d_std.get(),
        (const float *)d_pad.get());

  } else {
    cuda_op::crop_resize_normalize_gpu(
        (const uint8_t *)d_inputImage.get(),
        (float *)hwcBuffer.getRawDevicePtr(), image.rows, image.cols, src_c,
        roi.x, roi.y, src_h, src_w, args.modelInputShape.h,
        args.modelInputShape.w, (const float *)d_mean.get(),
        (const float *)d_std.get());
  }

  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);
  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  if (args.hwc2chw) {
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
