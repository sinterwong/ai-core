/**
 * @file trt_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __TRT_UTILS_HPP__
#define __TRT_UTILS_HPP__

#include "ai_core/infer_common_types.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <logger.hpp>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      LOG_ERRORS << "CUDA error in " << __FILE__ << " line " << __LINE__       \
                 << ": " << cudaGetErrorString(err) << " (" << err << ")";     \
      throw std::runtime_error("CUDA error: " +                                \
                               std::string(cudaGetErrorString(err)));          \
    }                                                                          \
  } while (0)

namespace ai_core::trt_utils {

// convert ai_core::DataType to nvinfer1::DataType
nvinfer1::DataType aiCoreDataTypeToTrt(ai_core::DataType type);

// convert nvinfer1::DataType to ai_core::DataType
ai_core::DataType trtDataTypeToAiCore(nvinfer1::DataType trt_type);

// get element size in bytes for nvinfer1::DataType
size_t getTrtElementSize(nvinfer1::DataType trt_type);

} // namespace ai_core::trt_utils

#endif // __TRT_UTILS_HPP__
