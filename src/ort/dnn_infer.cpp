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

  mInputNames.clear();
  mInputShapes.clear();
  mOutputNames.clear();
  mOutputShapes.clear();
  modelInfo.reset();

  try {
    LOG_INFOS << "Initializing model: " << mParams.name;

    // create environment
    mEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      mParams.name.c_str());

    // mSession options
    Ort::SessionOptions mSessionOptions;
    int threadNum = std::thread::hardware_concurrency();
    mSessionOptions.SetIntraOpNumThreads(threadNum);
    mSessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    mSessionOptions.SetDeterministicCompute(true);

    LOG_INFOS << "Creating mSession options for model: " << mParams.name;
    // create mSession
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
          *mEnv, adaPlatformPath(mParams.modelPath).c_str(), mSessionOptions);
    } else {
      mSession = std::make_unique<Ort::Session>(
          *mEnv, engineData.data(), engineData.size(), mSessionOptions);
    }

    // create memory info
    mMemoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // get input info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = mSession->GetInputCount();
    mInputNames.resize(numInputNodes);
    mInputShapes.resize(numInputNodes);

    for (size_t i = 0; i < numInputNodes; i++) {
      // get input name
      auto inputName = mSession->GetInputNameAllocated(i, allocator);
      mInputNames[i] = inputName.get();

      // get input shape
      auto typeInfo = mSession->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      mInputShapes[i] = tensorInfo.GetShape();
    }

    // get output info
    size_t numOutputNodes = mSession->GetOutputCount();
    mOutputNames.resize(numOutputNodes);
    mOutputShapes.resize(numOutputNodes);

    for (size_t i = 0; i < numOutputNodes; i++) {
      // get output name
      auto outputName = mSession->GetOutputNameAllocated(i, allocator);
      mOutputNames[i] = outputName.get();

      // get output shape
      auto typeInfo = mSession->GetOutputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      mOutputShapes[i] = tensorInfo.GetShape();
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
  if (mEnv == nullptr || mSession == nullptr || mMemoryInfo == nullptr) {
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

    std::vector<const char *> mInputNamesPtr;
    std::vector<const char *> mOutputNamesPtr;

    mInputNamesPtr.reserve(mInputNames.size());
    mOutputNamesPtr.reserve(mOutputNames.size());

    for (const auto &name : mInputNames) {
      mInputNamesPtr.push_back(name.c_str());
    }
    for (const auto &name : mOutputNames) {
      mOutputNamesPtr.push_back(name.c_str());
    }

    if (prepDatas.size() != mInputShapes.size()) {
      LOG_ERRORS << "Input data count (" << prepDatas.size()
                 << ") doesn't match input shapes count ("
                 << mInputShapes.size() << ")";
      return InferErrorCode::INFER_FAILED;
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(prepDatas.size());
    for (size_t i = 0; i < mInputShapes.size(); ++i) {
      const std::string &inputName = mInputNames.at(i);
      auto &prepData = prepDatas.at(inputName);
      switch (prepData.dataType()) {
      case DataType::FLOAT32: {
        inputs.emplace_back(Ort::Value::CreateTensor(
            *mMemoryInfo, const_cast<void *>(prepData.getRawHostPtr()),
            prepData.getSizeBytes(), mInputShapes[i].data(),
            mInputShapes[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        break;
      }

      case DataType::FLOAT16: {
#if ORT_API_VERSION >= 12
        inputs.emplace_back(Ort::Value::CreateTensor(
            *mMemoryInfo, const_cast<void *>(prepData.getRawHostPtr()),
            prepData.getSizeBytes(), mInputShapes[i].data(),
            mInputShapes[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
#else
        size_t elemCount = prepData.getElementCount();
        auto typedPtr = prepData.getHostPtr<uint16_t>();
        inputs.emplace_back(Ort::Value::CreateTensor<uint16_t>(
            *mMemoryInfo, const_cast<uint16_t *>(typedPtr), elemCount,
            mInputShapes[i].data(), mInputShapes[i].size()));
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
    // mSession.Run itself is thread-safe
    modelOutputs = mSession->Run(
        Ort::RunOptions{nullptr}, mInputNamesPtr.data(), inputs.data(),
        inputs.size(), mOutputNamesPtr.data(), mOutputNames.size());
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
          std::make_pair(mOutputNames.at(i), std::move(outputData)));
      std::vector<int> outputShape;
      outputShape.reserve(
          modelOutput.GetTensorTypeAndShapeInfo().GetShape().size());
      for (int64_t dim : modelOutput.GetTensorTypeAndShapeInfo().GetShape()) {
        outputShape.push_back(static_cast<int>(dim));
      }
      outputs.shapes.insert(std::make_pair(mOutputNames.at(i), outputShape));
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
    mSession.reset();
    mEnv.reset();
    mMemoryInfo.reset();

    mInputNames.clear();
    mInputShapes.clear();
    mOutputNames.clear();
    mOutputShapes.clear();

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

  modelInfo->name = mParams.name;
  if (!mSession) {
    LOG_ERRORS << "Session is not initialized";
    return *modelInfo;
  }
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = mSession->GetInputCount();
    modelInfo->inputs.resize(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++) {
      auto inputName = mSession->GetInputNameAllocated(i, allocator);
      modelInfo->inputs[i].name = inputName.get();

      auto typeInfo = mSession->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

      modelInfo->inputs[i].shape = tensorInfo.GetShape();

      size_t numOutputNodes = mSession->GetOutputCount();
      modelInfo->outputs.resize(numOutputNodes);

      for (size_t i = 0; i < numOutputNodes; i++) {
        auto outputName = mSession->GetOutputNameAllocated(i, allocator);
        modelInfo->outputs[i].name = outputName.get();
        auto typeInfo = mSession->GetOutputTypeInfo(i);
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
