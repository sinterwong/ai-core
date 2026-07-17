/**
 * @file ort_dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ort_infer.hpp"
#include "ai_core/logger.hpp"
#include "crypto.hpp"
#include <numeric>
#include <opencv2/opencv.hpp>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ai_core::dnn {

inline auto adaPlatformPath(const std::string &path) {
#ifdef _WIN32
  const int len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(),
                                      static_cast<int>(path.size()), nullptr, 0);
  std::wstring wpath(static_cast<size_t>(len), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(),
                      static_cast<int>(path.size()), wpath.data(), len);
  return wpath;
#else
  return path;
#endif
}

ONNXTensorElementDataType OrtAlgoInference::aiCoreDataTypeToOrt(DataType type) {
  switch (type) {
  case DataType::FLOAT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case DataType::FLOAT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case DataType::INT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case DataType::INT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case DataType::INT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  default:
    throw std::invalid_argument(
        "Unsupported or unknown DataType for ONNX Runtime.");
  }
}

DataType OrtAlgoInference::ortDataTypeToAiCore(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return DataType::FLOAT32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return DataType::FLOAT16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return DataType::INT64;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return DataType::INT32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return DataType::INT8;
  default:
    throw std::invalid_argument("Unsupported or unknown ONNX data type.");
  }
}

InferErrorCode OrtAlgoInference::initialize() {
  std::unique_lock lk(m_mutex);

  m_inputNames.clear();
  m_outputNames.clear();
  m_modelInfo.reset();

  try {
    LOG_INFO_S << "Initializing model: " << m_params.name;

    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                       m_params.name.c_str());

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::vector<unsigned char> engine_data;
    if (m_params.need_decrypt) {
      auto crypto_config =
          encrypt::Crypto::deriveKeyFromCommit(m_params.decryptkey_str);
      encrypt::Crypto crypto(crypto_config);
      if (!crypto.decryptData(m_params.model_path, engine_data)) {
        LOG_ERROR_S << "Failed to decrypt model data: " << m_params.model_path;
        return InferErrorCode::InitDecryptionFailed;
      }
      if (engine_data.empty()) {
        LOG_ERROR_S << "Decryption resulted in empty model data: "
                    << m_params.model_path;
        return InferErrorCode::InitModelLoadFailed;
      }
    }

    if (engine_data.empty()) {
      m_session = std::make_unique<Ort::Session>(
          *m_env, adaPlatformPath(m_params.model_path).c_str(),
          session_options);
    } else {
      m_session = std::make_unique<Ort::Session>(
          *m_env, engine_data.data(), engine_data.size(), session_options);
    }

    m_memoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // Build ModelInfo and store names
    Ort::AllocatorWithDefaultOptions allocator;
    m_modelInfo = std::make_shared<ModelInfo>();
    m_modelInfo->name = m_params.name;

    // Get input info
    size_t num_input_nodes = m_session->GetInputCount();
    m_inputNames.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
      auto input_name = m_session->GetInputNameAllocated(i, allocator);
      m_inputNames.emplace_back(input_name.get());

      auto type_info = m_session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      ModelInfo::TensorInfo ti;
      ti.name = input_name.get();
      // This will correctly have -1 for dynamic dims
      ti.shape = tensor_info.GetShape();
      ti.data_type = ortDataTypeToAiCore(tensor_info.GetElementType());

      for (const auto &dim : ti.shape) {
        if (dim < 0) {
          m_dynamicInputTensorNames.insert(ti.name);
          LOG_INFO_S << "Input tensor '" << ti.name
                     << "' is identified as dynamic.";
          break;
        }
      }

      m_modelInfo->inputs.push_back(std::move(ti));
    }

    // Get output info
    size_t num_output_nodes = m_session->GetOutputCount();
    m_outputNames.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
      auto output_name = m_session->GetOutputNameAllocated(i, allocator);
      m_outputNames.emplace_back(output_name.get());

      auto type_info = m_session->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      ModelInfo::TensorInfo ti;
      ti.name = output_name.get();
      ti.shape = tensor_info.GetShape();
      ti.data_type = ortDataTypeToAiCore(tensor_info.GetElementType());
      m_modelInfo->outputs.push_back(std::move(ti));
    }

    LOG_INFO_S << "Model " << m_params.name << " initialized successfully";
    return InferErrorCode::SUCCESS;

  } catch (const Ort::Exception &e) {
    LOG_ERROR_S << "ONNX Runtime error during initialization: " << e.what();
    return InferErrorCode::InitModelLoadFailed;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Error during initialization: " << e.what();
    return InferErrorCode::InitFailed;
  }
}

InferErrorCode OrtAlgoInference::infer(const TensorData &inputs,
                                       TensorData &outputs) {
  std::shared_lock lk(m_mutex);
  if (!m_session) {
    LOG_ERROR_S << "Session is not initialized";
    return InferErrorCode::NotInitialized;
  }

  try {
    outputs.datas.clear();
    outputs.shapes.clear();

    if (inputs.datas.size() != m_inputNames.size()) {
      LOG_ERROR_S << "Input data count (" << inputs.datas.size()
                  << ") doesn't match model's expected input count ("
                  << m_inputNames.size() << ")";
      return InferErrorCode::InferInputError;
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(m_inputNames.size());

    for (const auto &name : m_inputNames) {
      auto dataIt = inputs.datas.find(name);
      if (dataIt == inputs.datas.end()) {
        LOG_ERROR_S << "Input tensor '" << name
                    << "' not found in provided inputs.";
        return InferErrorCode::InferInputError;
      }
      const auto &input_buffer = dataIt->second;

      std::vector<int64_t> inputShape;
      const bool isDynamic = m_dynamicInputTensorNames.count(name);

      if (isDynamic) {
        auto shapeIt = inputs.shapes.find(name);
        if (shapeIt == inputs.shapes.end()) {
          LOG_ERROR_S << "Shape info for dynamic input tensor '" << name
                      << "' must be provided.";
          return InferErrorCode::InferInputError;
        }
        const auto &inputShapeVecInt = shapeIt->second;
        inputShape.assign(inputShapeVecInt.begin(), inputShapeVecInt.end());

      } else {
        const auto &modelInputInfo =
            std::find_if(m_modelInfo->inputs.begin(), m_modelInfo->inputs.end(),
                         [&](const ModelInfo::TensorInfo &info) {
                           return info.name == name;
                         });

        if (modelInputInfo != m_modelInfo->inputs.end()) {
          const auto &staticShapeVecInt = modelInputInfo->shape;
          inputShape.assign(staticShapeVecInt.begin(), staticShapeVecInt.end());
        } else {
          LOG_ERROR_S
              << "Internal error: Could not find static shape info for input '"
              << name << "'.";
          return InferErrorCode::InferFailed;
        }

        int64_t expectedVolume =
            std::accumulate(inputShape.begin(), inputShape.end(), 1LL,
                            std::multiplies<int64_t>());
        size_t elementSize =
            TypedBuffer::getElementSize(input_buffer.dataType());
        size_t expectedSizeBytes = expectedVolume * elementSize;
        if (input_buffer.getSizeBytes() != expectedSizeBytes) {
          LOG_ERROR_S << "Mismatched size for static input tensor '" << name
                      << "'. Expected: " << expectedSizeBytes
                      << " bytes, Got: " << input_buffer.getSizeBytes()
                      << " bytes.";
          return InferErrorCode::InferSizeMismatch;
        }
      }

      input_tensors.emplace_back(Ort::Value::CreateTensor(
          *m_memoryInfo, const_cast<void *>(input_buffer.getRawHostPtr()),
          input_buffer.getSizeBytes(), inputShape.data(), inputShape.size(),
          aiCoreDataTypeToOrt(input_buffer.dataType())));
    }

    std::vector<const char *> input_names_ptr;
    input_names_ptr.reserve(m_inputNames.size());
    for (const auto &name : m_inputNames) {
      input_names_ptr.push_back(name.c_str());
    }

    std::vector<const char *> output_names_ptr;
    output_names_ptr.reserve(m_outputNames.size());
    for (const auto &name : m_outputNames) {
      output_names_ptr.push_back(name.c_str());
    }

    auto output_tensors = m_session->Run(
        Ort::RunOptions{nullptr}, input_names_ptr.data(), input_tensors.data(),
        input_tensors.size(), output_names_ptr.data(), output_names_ptr.size());

    for (size_t i = 0; i < output_tensors.size(); ++i) {
      const auto &output_tensor = output_tensors[i];
      auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();

      // Get raw data and size
      const void *raw_data = output_tensor.GetTensorRawData();

      size_t element_count = tensor_info.GetElementCount();
      ONNXTensorElementDataType type = tensor_info.GetElementType();
      size_t element_size = 0;
      switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        element_size = sizeof(float);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        element_size = sizeof(uint16_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        element_size = sizeof(int64_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        element_size = sizeof(int32_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        element_size = sizeof(int8_t);
        break;
      default:
        LOG_ERROR_S << "Unsupported data type for size calculation.";
        return InferErrorCode::InferFailed;
      }
      size_t byte_size = element_count * element_size;

      // Create a copy of the output data
      std::vector<uint8_t> byte_data(static_cast<const uint8_t *>(raw_data),
                                     static_cast<const uint8_t *>(raw_data) +
                                         byte_size);

      DataType output_type = ortDataTypeToAiCore(tensor_info.GetElementType());
      TypedBuffer output_buffer =
          TypedBuffer::createFromCpu(output_type, std::move(byte_data));

      outputs.datas[m_outputNames[i]] = std::move(output_buffer);
      auto output_shape_int64 = tensor_info.GetShape();
      std::vector<int> output_shape(output_shape_int64.begin(),
                                    output_shape_int64.end());
      outputs.shapes[m_outputNames[i]] = std::move(output_shape);
    }

    return InferErrorCode::SUCCESS;

  } catch (const Ort::Exception &e) {
    LOG_ERROR_S << "ONNX Runtime error during inference: " << e.what();
    return InferErrorCode::InferFailed;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Error during inference: " << e.what();
    return InferErrorCode::InferFailed;
  }
}

InferErrorCode OrtAlgoInference::terminate() {
  std::unique_lock lk(m_mutex);
  m_session.reset();
  m_env.reset();
  m_memoryInfo.reset();
  m_inputNames.clear();
  m_outputNames.clear();
  m_modelInfo.reset();
  return InferErrorCode::SUCCESS;
}

const ModelInfo &OrtAlgoInference::getModelInfo() {
  std::shared_lock lk(m_mutex);
  if (!m_modelInfo) {
    LOG_WARNING_S << "getModelInfo() called on uninitialized or failed model.";
    // Return a static empty info to avoid null reference
    static ModelInfo empty_info;
    return empty_info;
  }
  return *m_modelInfo;
}
}; // namespace ai_core::dnn
