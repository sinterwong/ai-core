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

#include <thread>

#include "crypto.hpp"
#include "dnn_infer.hpp"
#include "logger.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

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

InferErrorCode OrtAlgoInference::initialize() {
  std::lock_guard lk = std::lock_guard(mtx_);

  inputNames.clear();
  inputShapes.clear();
  outputNames.clear();
  outputShapes.clear();
  modelInfo.reset();

  try {
    LOG_INFOS << "Initializing model: " << params_.name;

    // create environment
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                     params_.name.c_str());

    // session options
    Ort::SessionOptions sessionOptions;
    int threadNum = std::thread::hardware_concurrency();
    sessionOptions.SetIntraOpNumThreads(threadNum);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    sessionOptions.SetDeterministicCompute(true);

    LOG_INFOS << "Creating session options for model: " << params_.name;
    // create session
    std::vector<unsigned char> engineData;
    if (params_.needDecrypt) {
      auto cryptoConfig =
          encrypt::Crypto::deriveKeyFromCommit(params_.decryptkeyStr);
      encrypt::Crypto crypto(cryptoConfig);
      if (!crypto.decryptData(params_.modelPath, engineData)) {
        LOG_ERRORS << "Failed to decrypt model data: " << params_.modelPath;
        return InferErrorCode::INIT_DECRYPTION_FAILED;
      }
      if (engineData.empty()) {
        LOG_ERRORS << "Decryption resulted in empty model data: "
                   << params_.modelPath;
        return InferErrorCode::INIT_MODEL_LOAD_FAILED;
      }
    }

    if (engineData.empty()) {
      session = std::make_unique<Ort::Session>(
          *env, adaPlatformPath(params_.modelPath).c_str(), sessionOptions);
    } else {
      session = std::make_unique<Ort::Session>(
          *env, engineData.data(), engineData.size(), sessionOptions);
    }

    // create memory info
    memoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // get input info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    inputNames.resize(numInputNodes);
    inputShapes.resize(numInputNodes);

    for (size_t i = 0; i < numInputNodes; i++) {
      // get input name
      auto inputName = session->GetInputNameAllocated(i, allocator);
      inputNames[i] = inputName.get();

      // get input shape
      auto typeInfo = session->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      inputShapes[i] = tensorInfo.GetShape();
    }

    // get output info
    size_t numOutputNodes = session->GetOutputCount();
    outputNames.resize(numOutputNodes);
    outputShapes.resize(numOutputNodes);

    for (size_t i = 0; i < numOutputNodes; i++) {
      // get output name
      auto outputName = session->GetOutputNameAllocated(i, allocator);
      outputNames[i] = outputName.get();

      // get output shape
      auto typeInfo = session->GetOutputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      outputShapes[i] = tensorInfo.GetShape();
    }
    LOG_INFOS << "Model " << params_.name << " initialized successfully";
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
  if (env == nullptr || session == nullptr || memoryInfo == nullptr) {
    LOG_ERRORS << "Session is not initialized";
    return InferErrorCode::INFER_FAILED;
  }
  try {
    outputs.datas.clear();
    outputs.shapes.clear();

    const auto &prepDatas = inputs.datas;
    const auto &prepDatasShapes = inputs.shapes;

    if (prepDatas.empty()) {
      LOG_ERRORS << "Empty input data after preprocessing";
      return InferErrorCode::INFER_PREPROCESS_FAILED;
    }

    std::vector<const char *> inputNamesPtr;
    std::vector<const char *> outputNamesPtr;

    inputNamesPtr.reserve(inputNames.size());
    outputNamesPtr.reserve(outputNames.size());

    for (const auto &name : inputNames) {
      inputNamesPtr.push_back(name.c_str());
    }
    for (const auto &name : outputNames) {
      outputNamesPtr.push_back(name.c_str());
    }

    if (prepDatas.size() != inputShapes.size()) {
      LOG_ERRORS << "Input data count (" << prepDatas.size()
                 << ") doesn't match input shapes count (" << inputShapes.size()
                 << ")";
      return InferErrorCode::INFER_FAILED;
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(prepDatas.size());
    for (size_t i = 0; i < inputShapes.size(); ++i) {
      const std::string &inputName = inputNames.at(i);
      auto &prepData = prepDatas.at(inputName);
      switch (prepData.dataType()) {
      case DataType::FLOAT32: {
        inputs.emplace_back(Ort::Value::CreateTensor(
            *memoryInfo, const_cast<void *>(prepData.getRawHostPtr()),
            prepData.getSizeBytes(), inputShapes[i].data(),
            inputShapes[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        break;
      }

      case DataType::FLOAT16: {
#if ORT_API_VERSION >= 12
        inputs.emplace_back(Ort::Value::CreateTensor(
            *memoryInfo, const_cast<void *>(prepData.getRawHostPtr()),
            prepData.getSizeBytes(), inputShapes[i].data(),
            inputShapes[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
#else
        size_t elemCount = prepData.getElementCount();
        auto typedPtr = prepData.getHostPtr<uint16_t>();
        inputs.emplace_back(Ort::Value::CreateTensor<uint16_t>(
            *memoryInfo, const_cast<uint16_t *>(typedPtr), elemCount,
            inputShapes[i].data(), inputShapes[i].size()));
#endif
        break;
      }

      default:
        LOG_ERRORS << "Unsupported data type: "
                   << static_cast<int>(prepData.dataType());
        return InferErrorCode::INFER_FAILED;
      }
    }

    std::vector<Ort::Value> modelOutputs;
    auto inferStart = std::chrono::steady_clock::now();
    // session.Run itself is thread-safe
    modelOutputs = session->Run(Ort::RunOptions{nullptr}, inputNamesPtr.data(),
                                inputs.data(), inputs.size(),
                                outputNamesPtr.data(), outputNames.size());
    for (size_t i = 0; i < modelOutputs.size(); ++i) {
      auto &modelOutput = modelOutputs[i];
      auto typeInfo = modelOutput.GetTensorTypeAndShapeInfo();
      auto elemCount = typeInfo.GetElementCount();

      TypedBuffer outputData;
      auto elemType = typeInfo.GetElementType();
      if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const auto *rawData = modelOutput.GetTensorData<uint8_t>();
        const size_t byteSize = elemCount * sizeof(float);

        std::vector<uint8_t> byteData(rawData, rawData + byteSize);
        outputData =
            TypedBuffer::createFromCpu(DataType::FLOAT32, std::move(byteData));
      } else if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // FIXME: FP16 will be converted to FP32 right now
        // FP16 -> FP32
        const uint16_t *fp16Data = modelOutput.GetTensorData<uint16_t>();
        cv::Mat halfMat(1, elemCount, CV_16F, (void *)fp16Data);
        cv::Mat floatMat(1, elemCount, CV_32F);
        halfMat.convertTo(floatMat, CV_32F);

        const size_t byteSize = elemCount * sizeof(float);
        const auto *floatMatData =
            reinterpret_cast<const uint8_t *>(floatMat.data);
        std::vector<uint8_t> byteData(floatMatData, floatMatData + byteSize);
        outputData =
            TypedBuffer::createFromCpu(DataType::FLOAT32, std::move(byteData));
      } else {
        LOG_ERRORS << "Unsupported output tensor data type: "
                   << static_cast<int>(elemType);
        return InferErrorCode::INFER_FAILED;
      }

      outputs.datas.insert(
          std::make_pair(outputNames.at(i), std::move(outputData)));
      std::vector<int> outputShape;
      outputShape.reserve(
          modelOutput.GetTensorTypeAndShapeInfo().GetShape().size());
      for (int64_t dim : modelOutput.GetTensorTypeAndShapeInfo().GetShape()) {
        outputShape.push_back(static_cast<int>(dim));
      }
      outputs.shapes.insert(std::make_pair(outputNames.at(i), outputShape));
      auto inferEnd = std::chrono::steady_clock::now();
      auto durationInfer =
          std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd -
                                                                inferStart);
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
  std::lock_guard lk = std::lock_guard(mtx_);
  try {
    session.reset();
    env.reset();
    memoryInfo.reset();

    inputNames.clear();
    inputShapes.clear();
    outputNames.clear();
    outputShapes.clear();

    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Error during termination: " << e.what();
    return InferErrorCode::TERMINATE_FAILED;
  }
}

const ModelInfo &OrtAlgoInference::getModelInfo() {
  if (modelInfo)
    return *modelInfo;

  modelInfo = std::make_shared<ModelInfo>();

  modelInfo->name = params_.name;
  if (!session) {
    LOG_ERRORS << "Session is not initialized";
    return *modelInfo;
  }
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    modelInfo->inputs.resize(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++) {
      auto inputName = session->GetInputNameAllocated(i, allocator);
      modelInfo->inputs[i].name = inputName.get();

      auto typeInfo = session->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

      modelInfo->inputs[i].shape = tensorInfo.GetShape();

      size_t numOutputNodes = session->GetOutputCount();
      modelInfo->outputs.resize(numOutputNodes);

      for (size_t i = 0; i < numOutputNodes; i++) {
        auto outputName = session->GetOutputNameAllocated(i, allocator);
        modelInfo->outputs[i].name = outputName.get();
        auto typeInfo = session->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        modelInfo->outputs[i].shape = tensorInfo.GetShape();
      }
    }
  } catch (const Ort::Exception &e) {
    LOG_ERRORS << "ONNX Runtime error during getting model info: " << e.what();
  } catch (const std::exception &e) {
    LOG_ERRORS << "Error during getting model info: " << e.what();
  }
  return *modelInfo;
}
}; // namespace ai_core::dnn
