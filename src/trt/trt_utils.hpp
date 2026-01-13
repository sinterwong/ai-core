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
#ifndef AI_CORE_TRT_UTILS_HPP
#define AI_CORE_TRT_UTILS_HPP

#include "ai_core/common_types.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace ai_core::trt_utils {

// convert ai_core::DataType to nvinfer1::DataType
nvinfer1::DataType aiCoreDataTypeToTrt(ai_core::DataType type);

// convert nvinfer1::DataType to ai_core::DataType
ai_core::DataType trtDataTypeToAiCore(nvinfer1::DataType trt_type);

// get element size in bytes for nvinfer1::DataType
size_t getTrtElementSize(nvinfer1::DataType trt_type);

} // namespace ai_core::trt_utils

#endif
