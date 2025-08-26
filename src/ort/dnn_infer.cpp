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

#include "dnn_infer.hpp"
#include "crypto.hpp"
#include <logger.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <thread>

#ifdef _WIN32
#include <codecvt>
#endif

namespace ai_core::dnn {

inline auto adaPlatformPath(const std::string &path) {
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(path);
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
  std::lock_guard lk(mMutex);

  mInputNames.clear();
  mOutputNames.clear();
  modelInfo.reset();

  try {
    LOG_INFOS << "Initializing model: " << mParams.name;

    mEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      mParams.name.c_str());

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::vector<unsigned char> engineData;
    if (mParams.needDecrypt) {
      auto cryptoConfig =
          encrypt::Crypto::deriveKeyFromCommit(mParams.decryptkeyStr);
      encrypt::Crypto crypto(cryptoConfig);
      if (!crypto.decryptData(mParams.modelPath, engineData)) {
        LOG_ERRORS << "Failed to decrypt model data: " << mParams.modelPath;
        return InferErrorCode::INIT_DECRYPTION_FAILED;
      }
      if (engineData.empty()) {
        LOG_ERRORS << "Decryption resulted in empty model data: "
                   << mParams.modelPath;
        return InferErrorCode::INIT_MODEL_LOAD_FAILED;
      }
    }

    if (engineData.empty()) {
      mSession = std::make_unique<Ort::Session>(
          *mEnv, adaPlatformPath(mParams.modelPath).c_str(), sessionOptions);
    } else {
      mSession = std::make_unique<Ort::Session>(
          *mEnv, engineData.data(), engineData.size(), sessionOptions);
    }

    mMemoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // Build ModelInfo and store names
    Ort::AllocatorWithDefaultOptions allocator;
    modelInfo = std::make_shared<ModelInfo>();
    modelInfo->name = mParams.name;

    // Get input info
    size_t numInputNodes = mSession->GetInputCount();
    mInputNames.reserve(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++) {
      auto inputName = mSession->GetInputNameAllocated(i, allocator);
      mInputNames.emplace_back(inputName.get());

      auto typeInfo = mSession->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

      ModelInfo::TensorInfo ti;
      ti.name = inputName.get();
      // This will correctly have -1 for dynamic dims
      ti.shape = tensorInfo.GetShape();
      ti.dataType = ortDataTypeToAiCore(tensorInfo.GetElementType());
      modelInfo->inputs.push_back(std::move(ti));
    }

    // Get output info
    size_t numOutputNodes = mSession->GetOutputCount();
    mOutputNames.reserve(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++) {
      auto outputName = mSession->GetOutputNameAllocated(i, allocator);
      mOutputNames.emplace_back(outputName.get());

      auto typeInfo = mSession->GetOutputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

      ModelInfo::TensorInfo ti;
      ti.name = outputName.get();
      ti.shape = tensorInfo.GetShape();
      ti.dataType = ortDataTypeToAiCore(tensorInfo.GetElementType());
      modelInfo->outputs.push_back(std::move(ti));
    }

    LOG_INFOS << "Model " << mParams.name << " initialized successfully";
    return InferErrorCode::SUCCESS;

  } catch (const Ort::Exception &e) {
    LOG_ERRORS << "ONNX Runtime error during initialization: " << e.what();
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Error during initialization: " << e.what();
    return InferErrorCode::INIT_FAILED;
  }
}

InferErrorCode OrtAlgoInference::infer(const TensorData &inputs,
                                       TensorData &outputs) {
  if (!mSession) {
    LOG_ERRORS << "Session is not initialized";
    return InferErrorCode::NOT_INITIALIZED;
  }

  try {
    outputs.datas.clear();
    outputs.shapes.clear();

    if (inputs.datas.size() != mInputNames.size()) {
      LOG_ERRORS << "Input data count (" << inputs.datas.size()
                 << ") doesn't match model's expected input count ("
                 << mInputNames.size() << ")";
      return InferErrorCode::INFER_INPUT_ERROR;
    }

    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(mInputNames.size());

    for (const auto &name : mInputNames) {
      auto dataIt = inputs.datas.find(name);
      if (dataIt == inputs.datas.end()) {
        LOG_ERRORS << "Input tensor '" << name
                   << "' not found in provided inputs.";
        return InferErrorCode::INFER_INPUT_ERROR;
      }
      const auto &inputBuffer = dataIt->second;

      auto shapeIt = inputs.shapes.find(name);
      if (shapeIt == inputs.shapes.end()) {
        LOG_ERRORS << "Shape for input tensor '" << name << "' not found.";
        return InferErrorCode::INFER_INPUT_ERROR;
      }

      const auto &inputShapeVecInt = shapeIt->second;
      std::vector<int64_t> inputShape(inputShapeVecInt.begin(),
                                      inputShapeVecInt.end());

      inputTensors.emplace_back(Ort::Value::CreateTensor(
          *mMemoryInfo, const_cast<void *>(inputBuffer.getRawHostPtr()),
          inputBuffer.getSizeBytes(), inputShape.data(), inputShape.size(),
          aiCoreDataTypeToOrt(inputBuffer.dataType())));
    }

    std::vector<const char *> inputNamesPtr;
    inputNamesPtr.reserve(mInputNames.size());
    for (const auto &name : mInputNames) {
      inputNamesPtr.push_back(name.c_str());
    }

    std::vector<const char *> outputNamesPtr;
    outputNamesPtr.reserve(mOutputNames.size());
    for (const auto &name : mOutputNames) {
      outputNamesPtr.push_back(name.c_str());
    }

    auto outputTensors = mSession->Run(
        Ort::RunOptions{nullptr}, inputNamesPtr.data(), inputTensors.data(),
        inputTensors.size(), outputNamesPtr.data(), outputNamesPtr.size());

    for (size_t i = 0; i < outputTensors.size(); ++i) {
      const auto &outputTensor = outputTensors[i];
      auto tensorInfo = outputTensor.GetTensorTypeAndShapeInfo();

      // Get raw data and size
      const void *rawData = outputTensor.GetTensorRawData();

      size_t elementCount = tensorInfo.GetElementCount();
      ONNXTensorElementDataType type = tensorInfo.GetElementType();
      size_t elementSize = 0;
      switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        elementSize = sizeof(float);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        elementSize = sizeof(uint16_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        elementSize = sizeof(int64_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        elementSize = sizeof(int32_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        elementSize = sizeof(int8_t);
        break;
      default:
        LOG_ERRORS << "Unsupported data type for size calculation.";
        return InferErrorCode::INFER_FAILED;
      }
      size_t byteSize = elementCount * elementSize;

      // Create a copy of the output data
      std::vector<uint8_t> byteData(static_cast<const uint8_t *>(rawData),
                                    static_cast<const uint8_t *>(rawData) +
                                        byteSize);

      DataType outputType = ortDataTypeToAiCore(tensorInfo.GetElementType());
      TypedBuffer outputBuffer =
          TypedBuffer::createFromCpu(outputType, std::move(byteData));

      outputs.datas[mOutputNames[i]] = std::move(outputBuffer);
      auto outputShapeInt64 = tensorInfo.GetShape();
      std::vector<int> outputShape(outputShapeInt64.begin(),
                                   outputShapeInt64.end());
      outputs.shapes[mOutputNames[i]] = std::move(outputShape);
    }

    return InferErrorCode::SUCCESS;

  } catch (const Ort::Exception &e) {
    LOG_ERRORS << "ONNX Runtime error during inference: " << e.what();
    return InferErrorCode::INFER_FAILED;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Error during inference: " << e.what();
    return InferErrorCode::INFER_FAILED;
  }
}

InferErrorCode OrtAlgoInference::terminate() {
  std::lock_guard lk(mMutex);
  mSession.reset();
  mEnv.reset();
  mMemoryInfo.reset();
  mInputNames.clear();
  mOutputNames.clear();
  modelInfo.reset();
  return InferErrorCode::SUCCESS;
}

const ModelInfo &OrtAlgoInference::getModelInfo() {
  if (!modelInfo) {
    LOG_WARNINGS << "getModelInfo() called on uninitialized or failed model.";
    // Return a static empty info to avoid null reference
    static ModelInfo emptyInfo;
    return emptyInfo;
  }
  return *modelInfo;
}
}; // namespace ai_core::dnn
