/**
 * @file dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "trt_infer.hpp"
#include "ai_core/infer_error_code.hpp"
#include "crypto.hpp"
#include "trt_infer_stream.hpp"
#include "trt_utils.hpp"
#include <filesystem>
#include <fstream>
#include <numeric>

namespace ai_core::dnn {

TrtAlgoInference::TrtAlgoInference(const AlgoConstructParams &params)
    : mParams(params.getParam<AlgoInferParams>("params")) {
  LOG_INFO_S << "TrtAlgoInference created for model: " << mParams.name;
}

TrtAlgoInference::~TrtAlgoInference() { terminate(); }

// ============================================================================
// IInferEnginePlugin Implementation
// ============================================================================

InferErrorCode TrtAlgoInference::initialize() {
  std::lock_guard<std::mutex> lock(mMutex);
  if (mIsInitialized) {
    LOG_INFO_S << "TrtAlgoInference already initialized for model: "
               << mParams.name;
    return InferErrorCode::SUCCESS;
  }

  LOG_INFO_S << "Initializing TrtAlgoInference for model: " << mParams.name;

  try {
    InferErrorCode err =
        loadEngineFromPath(mParams.modelPath, mParams.needDecrypt);
    if (err != InferErrorCode::SUCCESS) {
      releaseResources();
      return err;
    }

    err = setupBindings();
    if (err != InferErrorCode::SUCCESS) {
      releaseResources();
      return err;
    }

    err = setupPinnedOutputBuffers();
    if (err != InferErrorCode::SUCCESS) {
      releaseResources();
      return err;
    }

    mIsInitialized = true;
    LOG_INFO_S << "TrtAlgoInference initialized successfully for model: "
               << mParams.name;
    LOG_INFO_S << "All inputs static: " << (mAllInputsStatic ? "yes" : "no");
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during initialization: " << e.what();
    releaseResources();
    return InferErrorCode::INIT_FAILED;
  }
}

InferErrorCode TrtAlgoInference::infer(const TensorData &inputs,
                                       TensorData &outputs) {
  std::lock_guard<std::mutex> lock(mMutex);

  if (!mIsInitialized) {
    LOG_ERROR_S << "Inference called on uninitialized model.";
    return InferErrorCode::NOT_INITIALIZED;
  }

  try {
    // Validate inputs
    for (const auto &inputInfo : modelInfo->inputs) {
      const auto &name = inputInfo.name;
      auto dataIt = inputs.datas.find(name);
      if (dataIt == inputs.datas.end()) {
        LOG_ERROR_S << "Input tensor '" << name
                    << "' not found in provided inputs.";
        return InferErrorCode::INFER_INPUT_ERROR;
      }

      const TypedBuffer &inputBuffer = dataIt->second;
      if (inputBuffer.dataType() != inputInfo.dataType) {
        LOG_ERROR_S << "Mismatched data type for input tensor '" << name
                    << "'. Expected: " << static_cast<int>(inputInfo.dataType)
                    << ", Got: " << static_cast<int>(inputBuffer.dataType());
        return InferErrorCode::INFER_TYPE_MISMATCH;
      }

      const size_t actualSizeBytes = inputBuffer.getSizeBytes();
      const bool isDynamic = mDynamicInputTensorNames.count(name);

      if (isDynamic) {
        auto shapeIt = inputs.shapes.find(name);
        if (shapeIt == inputs.shapes.end()) {
          LOG_ERROR_S << "Shape info for dynamic input tensor '" << name
                      << "' must be provided.";
          return InferErrorCode::INFER_INPUT_ERROR;
        }
        if (actualSizeBytes > mTensorSizeMap.at(name)) {
          LOG_ERROR_S << "Actual size for dynamic input '" << name
                      << "' exceeds max buffer size.";
          return InferErrorCode::INFER_SIZE_MISMATCH;
        }
      } else {
        if (actualSizeBytes != mTensorSizeMap.at(name)) {
          LOG_ERROR_S << "Mismatched size for static input tensor '" << name
                      << "'. Expected: " << mTensorSizeMap.at(name)
                      << " bytes, Got: " << actualSizeBytes << " bytes.";
          return InferErrorCode::INFER_SIZE_MISMATCH;
        }
      }

      if (inputBuffer.location() != BufferLocation::CPU &&
          inputBuffer.location() != BufferLocation::GPU_DEVICE) {
        LOG_ERROR_S << "Unsupported buffer location for input tensor: " << name;
        return InferErrorCode::INFER_INVALID_INPUT;
      }
    }

    if (!updateInputShapesIfNeeded(inputs)) {
      return InferErrorCode::INFER_EXECUTION_FAILED;
    }

    // Always use inferWithoutGraph for backward compatible sync mode
    return inferWithoutGraph(inputs, outputs);

  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during inference: " << e.what();
    CHECK_CUDA_ERROR(cudaStreamSynchronize(mStream));
    return InferErrorCode::INFER_FAILED;
  }
}

const ModelInfo &TrtAlgoInference::getModelInfo() {
  if (!mIsInitialized || !modelInfo) {
    LOG_WARNING_S << "getModelInfo() called on uninitialized model.";
    static ModelInfo emptyInfo;
    return emptyInfo;
  }
  return *modelInfo;
}

InferErrorCode TrtAlgoInference::terminate() {
  std::lock_guard<std::mutex> lock(mMutex);
  if (!mIsInitialized) {
    LOG_INFO_S
        << "TrtAlgoInference terminate called on uninitialized instance: "
        << mParams.name;
    return InferErrorCode::SUCCESS;
  }
  releaseResources();
  mIsInitialized = false;
  return InferErrorCode::SUCCESS;
}

// ============================================================================
// IAsyncInferEngine Implementation
// ============================================================================

std::shared_ptr<IInferStream> TrtAlgoInference::createStream() {
  if (!mIsInitialized) {
    throw std::runtime_error("Cannot create stream: engine not initialized");
  }

  auto stream = std::make_shared<TrtInferStream>(*this);
  auto result = stream->initialize();
  if (result != InferErrorCode::SUCCESS) {
    throw std::runtime_error("Failed to initialize inference stream");
  }

  LOG_INFO_S << "Created new inference stream for model: " << mParams.name;
  return stream;
}

TypedBuffer TrtAlgoInference::allocatePinnedHostBuffer(DataType type,
                                                       size_t sizeBytes) {
  return TypedBuffer::createPinnedHost(type, sizeBytes);
}

TrtAlgoInference::StreamContext TrtAlgoInference::createStreamContext() {
  if (!mIsInitialized || !modelInfo) {
    throw std::runtime_error(
        "Cannot create stream context: engine not initialized");
  }

  StreamContext ctx;
  ctx.stream = createStream();

  // Pre-allocate pinned input buffers based on max sizes
  for (const auto &input : modelInfo->inputs) {
    size_t sizeBytes = mTensorSizeMap.at(input.name);
    ctx.pinnedInputs.datas[input.name] =
        TypedBuffer::createPinnedHost(input.dataType, sizeBytes);

    std::vector<int> shapeInt(input.shape.begin(), input.shape.end());
    ctx.pinnedInputs.shapes[input.name] = std::move(shapeInt);
  }

  // Pre-allocate pinned output buffers based on max sizes
  for (const auto &output : modelInfo->outputs) {
    size_t sizeBytes = mTensorSizeMap.at(output.name);
    ctx.pinnedOutputs.datas[output.name] =
        TypedBuffer::createPinnedHost(output.dataType, sizeBytes);

    std::vector<int> shapeInt(output.shape.begin(), output.shape.end());
    ctx.pinnedOutputs.shapes[output.name] = std::move(shapeInt);
  }

  LOG_INFO_S << "Created stream context with pre-allocated buffers";
  return ctx;
}

// ============================================================================
// Internal Implementation
// ============================================================================

void TrtAlgoInference::releaseResources() {
  LOG_INFO_S << "Releasing TensorRT resources for model: " << mParams.name;

  mContext.reset();
  mEngine.reset();
  mRuntime.reset();

  if (mStream) {
    cudaError_t err = cudaStreamDestroy(mStream);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG_WARNING_S << "Failed to destroy CUDA stream: "
                    << cudaGetErrorString(err);
    }
    mStream = nullptr;
  }

  mManagedBuffers.clear();
  mPinnedOutputBuffers.clear();
  mTensorAddressMap.clear();
  mTensorSizeMap.clear();
  mCachedInputShapes.clear();
  modelInfo.reset();

  LOG_INFO_S << "TensorRT resources released for model: " << mParams.name;
}

int64_t TrtAlgoInference::calculateVolume(const nvinfer1::Dims &dims) {
  return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1},
                         std::multiplies<int64_t>());
}

InferErrorCode TrtAlgoInference::loadEngineFromPath(const std::string &path,
                                                    bool needsDecrypt) {
  if (!std::filesystem::exists(path)) {
    LOG_ERROR_S << "Model file does not exist: " << path;
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }

  std::vector<char> engineData;
  if (needsDecrypt) {
    LOG_INFO_S << "Decrypting TensorRT engine: " << path;
    std::vector<unsigned char> decryptedData;
    auto cryptoConfig =
        encrypt::Crypto::deriveKeyFromCommit(mParams.decryptkeyStr);
    encrypt::Crypto crypto(cryptoConfig);
    if (!crypto.decryptData(path, decryptedData)) {
      LOG_ERROR_S << "Failed to decrypt model data: " << path;
      return InferErrorCode::INIT_DECRYPTION_FAILED;
    }
    if (decryptedData.empty()) {
      LOG_ERROR_S << "Decryption resulted in empty model data: " << path;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    engineData.assign(decryptedData.begin(), decryptedData.end());
  } else {
    std::ifstream engineFile(path, std::ios::binary);
    if (!engineFile) {
      LOG_ERROR_S << "Failed to open TensorRT engine file: " << path;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    engineData.resize(fileSize);
    engineFile.read(engineData.data(), fileSize);
  }

  if (engineData.empty()) {
    LOG_ERROR_S << "Engine data is empty for model: " << path;
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }

  mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
  if (!mRuntime) {
    LOG_ERROR_S << "Failed to create TensorRT Runtime.";
    return InferErrorCode::INIT_RUNTIME_FAILED;
  }

  mEngine.reset(
      mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
  if (!mEngine) {
    LOG_ERROR_S << "Failed to deserialize TensorRT engine.";
    return InferErrorCode::INIT_ENGINE_FAILED;
  }

  LOG_INFO_S << "TensorRT engine loaded and deserialized successfully: "
             << mParams.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::setupBindings() {
  mContext.reset(mEngine->createExecutionContext());
  if (!mContext) {
    LOG_ERROR_S << "Failed to create TensorRT Execution Context.";
    return InferErrorCode::INIT_CONTEXT_FAILED;
  }

  int leastPriority, greatestPriority;
  CHECK_CUDA_ERROR(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&mStream, cudaStreamNonBlocking,
                                                greatestPriority));

  mManagedBuffers.clear();
  mTensorAddressMap.clear();
  mTensorSizeMap.clear();
  mDynamicInputTensorNames.clear();
  mCachedInputShapes.clear();
  mAllInputsStatic = true;

  modelInfo = std::make_shared<ModelInfo>();
  modelInfo->name = mParams.name;

  const int profileIndex = 0;
  if (mEngine->getNbOptimizationProfiles() <= profileIndex) {
    LOG_ERROR_S << "Engine does not have optimization profile at index "
                << profileIndex;
    return InferErrorCode::INIT_FAILED;
  }

  const int32_t numIOTensors = mEngine->getNbIOTensors();

  // Set input shapes to MAX for buffer allocation
  for (int32_t i = 0; i < numIOTensors; ++i) {
    const char *name = mEngine->getIOTensorName(i);
    if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      auto maxDims = mEngine->getProfileShape(
          name, profileIndex, nvinfer1::OptProfileSelector::kMAX);
      if (!mContext->setInputShape(name, maxDims)) {
        LOG_WARNING_S
            << "Failed to set max input shape for auto-sizing tensor: " << name;
      }
    }
  }

  mManagedBuffers.reserve(numIOTensors);

  for (int32_t i = 0; i < numIOTensors; ++i) {
    const char *name = mEngine->getIOTensorName(i);
    auto trtDtype = mEngine->getTensorDataType(name);

    auto dims = mEngine->getProfileShape(name, profileIndex,
                                         nvinfer1::OptProfileSelector::kMAX);

    int64_t volume = -1;
    size_t bufferSize = 0;

    if (dims.nbDims >= 0) {
      volume = calculateVolume(dims);
    }

    if (volume < 0) {
      if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        auto it = mParams.maxOutputBufferSizes.find(name);
        if (it != mParams.maxOutputBufferSizes.end()) {
          bufferSize = it->second;
        } else {
          nvinfer1::Dims deducedDims = mContext->getTensorShape(name);
          int64_t deducedVolume = calculateVolume(deducedDims);

          if (deducedVolume > 0) {
            bufferSize = static_cast<size_t>(deducedVolume) *
                         trt_utils::getTrtElementSize(trtDtype);
          } else {
            LOG_ERROR_S << "Could not deduce max size for dynamic output: "
                        << name;
            return InferErrorCode::INIT_BINDING_FAILED;
          }
        }
      } else {
        LOG_ERROR_S << "Input tensor '" << name
                    << "' has unexpected dynamic dimension.";
        return InferErrorCode::INIT_BINDING_FAILED;
      }
    } else {
      bufferSize =
          static_cast<size_t>(volume) * trt_utils::getTrtElementSize(trtDtype);
    }

    mManagedBuffers.emplace_back(cuda_utils::DeviceByteBuffer{bufferSize});
    void *devicePtr = mManagedBuffers.back().unsafePtr();

    mTensorAddressMap[name] = devicePtr;
    mTensorSizeMap[name] = bufferSize;

    if (!mContext->setTensorAddress(name, devicePtr)) {
      LOG_ERROR_S << "Failed to set tensor address for: " << name;
      return InferErrorCode::INIT_BINDING_FAILED;
    }

    // Populate ModelInfo
    ModelInfo::TensorInfo tensorInfo;
    tensorInfo.name = name;

    auto bindingDims = mEngine->getTensorShape(name);
    tensorInfo.shape.assign(bindingDims.d, bindingDims.d + bindingDims.nbDims);
    tensorInfo.dataType = trt_utils::trtDataTypeToAiCore(trtDtype);

    if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      for (const auto &dim : tensorInfo.shape) {
        if (dim == -1) {
          mDynamicInputTensorNames.insert(name);
          mAllInputsStatic = false;
          LOG_DEBUG_S << "Input tensor '" << name
                      << "' is identified as dynamic.";
          break;
        }
      }
      modelInfo->inputs.emplace_back(std::move(tensorInfo));
    } else {
      modelInfo->outputs.emplace_back(std::move(tensorInfo));
    }
  }

  LOG_INFO_S << "Bindings and buffers configured for model: " << mParams.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::setupPinnedOutputBuffers() {
  for (const auto &outputInfo : modelInfo->outputs) {
    const auto &name = outputInfo.name;
    size_t bufferSize = mTensorSizeMap.at(name);

    cuda_utils::CudaHostBuffer<uint8_t> pinnedBuffer;
    pinnedBuffer.reserve(bufferSize);

    mPinnedOutputBuffers[name] = std::move(pinnedBuffer);

    LOG_DEBUG_S << "Pre-allocated " << bufferSize
                << " bytes of pinned memory for output: " << name;
  }

  LOG_INFO_S << "Pinned output buffers allocated for "
             << modelInfo->outputs.size() << " outputs.";
  return InferErrorCode::SUCCESS;
}

bool TrtAlgoInference::updateInputShapesIfNeeded(const TensorData &inputs) {
  for (const auto &inputInfo : modelInfo->inputs) {
    const auto &name = inputInfo.name;
    const bool isDynamic = mDynamicInputTensorNames.count(name);

    if (!isDynamic) {
      continue;
    }

    auto shapeIt = inputs.shapes.find(name);
    if (shapeIt == inputs.shapes.end()) {
      continue;
    }

    const std::vector<int64_t> newShape(shapeIt->second.begin(),
                                        shapeIt->second.end());
    auto cacheIt = mCachedInputShapes.find(name);

    if (cacheIt == mCachedInputShapes.end() || cacheIt->second != newShape) {
      nvinfer1::Dims actualDims;
      actualDims.nbDims = newShape.size();
      std::copy(newShape.begin(), newShape.end(), actualDims.d);

      if (!mContext->setInputShape(name.c_str(), actualDims)) {
        LOG_ERROR_S << "Failed to set input shape for tensor: " << name;
        return false;
      }

      mCachedInputShapes[name] = newShape;
      LOG_TRACE_S << "Updated input shape for tensor '" << name << "'";
    }
  }

  return true;
}

void TrtAlgoInference::copyInputsToDevice(const TensorData &inputs) {
  for (const auto &inputInfo : modelInfo->inputs) {
    const auto &name = inputInfo.name;
    const TypedBuffer &inputBuffer = inputs.datas.at(name);
    const size_t actualSizeBytes = inputBuffer.getSizeBytes();
    void *destDevicePtr = mTensorAddressMap.at(name);

    if (inputBuffer.location() == BufferLocation::CPU) {
      const void *srcHostPtr = inputBuffer.getRawHostPtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(destDevicePtr, srcHostPtr,
                                       actualSizeBytes, cudaMemcpyHostToDevice,
                                       mStream));
    } else if (inputBuffer.location() == BufferLocation::GPU_DEVICE) {
      void *srcDevicePtr = inputBuffer.getRawDevicePtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(destDevicePtr, srcDevicePtr,
                                       actualSizeBytes,
                                       cudaMemcpyDeviceToDevice, mStream));
    }
  }
}

void TrtAlgoInference::copyOutputsToHost(TensorData &outputs) {
  outputs.datas.clear();
  outputs.shapes.clear();

  for (const auto &outputInfo : modelInfo->outputs) {
    const auto &name = outputInfo.name;
    void *srcDevicePtr = mTensorAddressMap.at(name);

    nvinfer1::Dims actualOutputDims = mContext->getTensorShape(name.c_str());
    int64_t actualVolume = calculateVolume(actualOutputDims);

    size_t actualOutputSizeBytes =
        static_cast<size_t>(actualVolume) *
        trt_utils::getTrtElementSize(
            trt_utils::aiCoreDataTypeToTrt(outputInfo.dataType));

    auto &pinnedBuffer = mPinnedOutputBuffers.at(name);

    if (pinnedBuffer.capacity() < actualOutputSizeBytes) {
      pinnedBuffer.reserve(actualOutputSizeBytes);
    }

    uint8_t *destHostPtr = pinnedBuffer.writePtr(actualOutputSizeBytes);

    CHECK_CUDA_ERROR(cudaMemcpyAsync(destHostPtr, srcDevicePtr,
                                     actualOutputSizeBytes,
                                     cudaMemcpyDeviceToHost, mStream));

    outputs.shapes[name].assign(actualOutputDims.d,
                                actualOutputDims.d + actualOutputDims.nbDims);
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(mStream));

  for (const auto &outputInfo : modelInfo->outputs) {
    const auto &name = outputInfo.name;
    auto &pinnedBuffer = mPinnedOutputBuffers.at(name);
    std::vector<uint8_t> safeData = pinnedBuffer.toVector();
    outputs.datas[name] =
        TypedBuffer::createFromCpu(outputInfo.dataType, std::move(safeData));
  }
}

InferErrorCode TrtAlgoInference::inferWithoutGraph(const TensorData &inputs,
                                                   TensorData &outputs) {
  copyInputsToDevice(inputs);

  LOG_TRACE_S << "Executing inference on stream " << mStream;
  if (!mContext->enqueueV3(mStream)) {
    LOG_ERROR_S << "Failed to enqueue TensorRT inference.";
    return InferErrorCode::INFER_EXECUTION_FAILED;
  }

  copyOutputsToHost(outputs);

  return InferErrorCode::SUCCESS;
}

} // namespace ai_core::dnn