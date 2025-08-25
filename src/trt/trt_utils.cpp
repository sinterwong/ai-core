/**
 * @file trt_utils.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "trt_utils.hpp"

namespace ai_core::trt_utils {

nvinfer1::DataType aiCoreDataTypeToTrt(ai_core::DataType type) {
  switch (type) {
  case ai_core::DataType::FLOAT32:
    return nvinfer1::DataType::kFLOAT;
  case ai_core::DataType::FLOAT16:
    return nvinfer1::DataType::kHALF;
  case ai_core::DataType::INT32:
    return nvinfer1::DataType::kINT32;
  case ai_core::DataType::INT64:
    return nvinfer1::DataType::kINT64;
  case ai_core::DataType::INT8:
    return nvinfer1::DataType::kINT8;
  default:
    LOG_ERRORS << "Unsupported ai_core::DataType: " << static_cast<int>(type);
    throw std::runtime_error(
        "Unsupported ai_core::DataType for TensorRT conversion.");
  }
}

ai_core::DataType trtDataTypeToAiCore(nvinfer1::DataType trt_type) {
  switch (trt_type) {
  case nvinfer1::DataType::kFLOAT:
    return ai_core::DataType::FLOAT32;
  case nvinfer1::DataType::kHALF:
    return ai_core::DataType::FLOAT16;
  case nvinfer1::DataType::kINT32:
    return ai_core::DataType::INT32;
  case nvinfer1::DataType::kINT64:
    return ai_core::DataType::INT64;
  case nvinfer1::DataType::kINT8:
    return ai_core::DataType::INT8;
  default:
    LOG_ERRORS << "Unsupported nvinfer1::DataType: "
               << static_cast<int>(trt_type);
    throw std::runtime_error(
        "Unsupported nvinfer1::DataType for ai_core conversion.");
  }
}

size_t getTrtElementSize(nvinfer1::DataType trt_type) {
  switch (trt_type) {
  case nvinfer1::DataType::kFLOAT:
    return sizeof(float);
  case nvinfer1::DataType::kHALF:
    return sizeof(uint16_t);
  case nvinfer1::DataType::kINT32:
    return sizeof(int32_t);
  case nvinfer1::DataType::kINT64:
    return sizeof(int64_t);
  case nvinfer1::DataType::kINT8:
    return sizeof(int8_t);
  case nvinfer1::DataType::kBOOL:
    return sizeof(bool);
  default:
    LOG_ERRORS << "Cannot get element size for unsupported nvinfer1::DataType: "
               << static_cast<int>(trt_type);
    throw std::runtime_error(
        "Unsupported nvinfer1::DataType for GetElementSize.");
  }
}
} // namespace ai_core::trt_utils
