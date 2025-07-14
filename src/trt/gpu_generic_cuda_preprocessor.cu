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
#include <logger.hpp>
#include <opencv2/core.hpp>

namespace ai_core::dnn::gpu {

TypedBuffer GpuGenericCudaPreprocessor::process(FramePreprocessArg &args,
                                                const FrameInput &input) const {
  const auto &image = *input.image;
  const auto &roi = *args.roi;

  // Allocate device memory for the input image
  trt_utils::TrtDeviceBuffer d_inputImage(image.total() * image.elemSize());
  CHECK_CUDA(cudaMemcpy(d_inputImage.get(), image.data,
                        image.total() * image.elemSize(),
                        cudaMemcpyHostToDevice));

  // Allocate device memory for mean and std
  trt_utils::TrtDeviceBuffer d_mean(args.meanVals.size() * sizeof(float));
  trt_utils::TrtDeviceBuffer d_std(args.normVals.size() * sizeof(float));
  CHECK_CUDA(cudaMemcpy(d_mean.get(), args.meanVals.data(),
                        args.meanVals.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_std.get(), args.normVals.data(),
                        args.normVals.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Allocate buffer for the preprocessed data on the GPU
  size_t totalElements = (size_t)args.modelInputShape.c *
                         args.modelInputShape.h * args.modelInputShape.w;
  size_t byteSize =
      totalElements * TypedBuffer::getElementSize(DataType::FLOAT32);
  TypedBuffer deviceBuffer =
      TypedBuffer::createFromGpu(DataType::FLOAT32, byteSize);

  if (args.isEqualScale) {
    trt_utils::TrtDeviceBuffer d_pad(args.pad.size() * sizeof(int));
    CHECK_CUDA(cudaMemcpy(d_pad.get(), args.pad.data(),
                          args.pad.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    cuda_op::escale_resize_normalize_gpu(
        (const uint8_t *)d_inputImage.get(),
        (float *)deviceBuffer.getRawDevicePtr(), image.rows, image.cols,
        image.channels(), args.modelInputShape.h, args.modelInputShape.w,
        (const float *)d_mean.get(), (const float *)d_std.get(),
        (const float *)d_pad.get());
  } else {
    cuda_op::crop_resize_normalize_gpu(
        (const uint8_t *)d_inputImage.get(),
        (float *)deviceBuffer.getRawDevicePtr(), image.rows, image.cols,
        image.channels(), roi.x, roi.y, roi.height, roi.width,
        args.modelInputShape.h, args.modelInputShape.w,
        (const float *)d_mean.get(), (const float *)d_std.get());
  }

  // Allocate final buffer based on data type
  size_t finalByteSize =
      totalElements * TypedBuffer::getElementSize(args.dataType);
  TypedBuffer finalDeviceBuffer =
      TypedBuffer::createFromGpu(args.dataType, finalByteSize);

  if (args.hwc2chw) {
    TypedBuffer chwBuffer =
        TypedBuffer::createFromGpu(DataType::FLOAT32, byteSize);
    cuda_op::hwc_to_chw_gpu((const float *)deviceBuffer.getRawDevicePtr(),
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
      cuda_op::fp32_to_fp16_gpu((const float *)deviceBuffer.getRawDevicePtr(),
                                (uint16_t *)finalDeviceBuffer.getRawDevicePtr(),
                                totalElements);
    } else {
      CHECK_CUDA(cudaMemcpy(finalDeviceBuffer.getRawDevicePtr(),
                            deviceBuffer.getRawDevicePtr(), finalByteSize,
                            cudaMemcpyDeviceToDevice));
    }
  }

  // Create the output TypedBuffer
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
