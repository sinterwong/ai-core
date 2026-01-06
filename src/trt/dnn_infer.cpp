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

#include "dnn_infer.hpp"
#include "ai_core/infer_error_code.hpp"
#include "crypto.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_host_buffer.cuh"
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

void TrtAlgoInference::releaseResources() {
  LOG_INFO_S << "Releasing TensorRT resources for model: " << mParams.name;

  mGraphCaptureInProgress.store(false, std::memory_order_release);
  if (mCudaGraphExec) {
    cudaGraphExecDestroy(mCudaGraphExec);
    mCudaGraphExec = nullptr;
  }
  if (mCudaGraph) {
    cudaGraphDestroy(mCudaGraph);
    mCudaGraph = nullptr;
  }
  mCudaGraphEnabled = false;
  mCudaGraphCaptured = false;

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

  // CudaDeviceBuffer 和 CudaHostBuffer 会自动释放内存
  mManagedBuffers.clear();
  mPinnedOutputBuffers.clear();
  mTensorAddressMap.clear();
  mTensorSizeMap.clear();
  mCachedInputShapes.clear();
  modelInfo.reset();

  LOG_INFO_S << "TensorRT resources released for model: " << mParams.name;
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
  LOG_INFO_S << "Using optimization profile 0.";

  const int32_t numIOTensors = mEngine->getNbIOTensors();

  // Set Input Shapes to MAX to allow automatic output size deduction
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

  // Allocate Buffers
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
          LOG_INFO_S << "Output tensor '" << name
                     << "' is dynamic. Using user-configured max buffer size: "
                     << bufferSize << " bytes.";
        } else {
          LOG_INFO_S << "Output tensor '" << name
                     << "' is dynamic and no user config found. "
                     << "Attempting to deduce max shape from context...";

          nvinfer1::Dims deducedDims = mContext->getTensorShape(name);
          int64_t deducedVolume = calculateVolume(deducedDims);

          if (deducedVolume > 0) {
            bufferSize = static_cast<size_t>(deducedVolume) *
                         trt_utils::getTrtElementSize(trtDtype);
            LOG_INFO_S << "Auto-deduced max buffer size for '" << name
                       << "': " << bufferSize << " bytes (" << deducedVolume
                       << " elements).";
          } else {
            LOG_ERROR_S
                << "Could not deduce max size for dynamic output tensor '"
                << name
                << "'. The shape might be data-dependent (value-dependent). "
                << "Please provide it in "
                   "AlgoInferParams::maxOutputBufferSizes.";
            return InferErrorCode::INIT_BINDING_FAILED;
          }
        }
      } else {
        LOG_ERROR_S << "Input tensor '" << name
                    << "' has an unexpected dynamic dimension (-1) in Profile.";
        return InferErrorCode::INIT_BINDING_FAILED;
      }
    } else {
      bufferSize =
          static_cast<size_t>(volume) * trt_utils::getTrtElementSize(trtDtype);
    }

    if (bufferSize == 0) {
      LOG_WARNING_S << "Tensor '" << name << "' has a buffer size of 0.";
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
  LOG_INFO_S << "All inputs static: " << (mAllInputsStatic ? "yes" : "no");

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::setupPinnedOutputBuffers() {
  for (const auto &outputInfo : modelInfo->outputs) {
    const auto &name = outputInfo.name;
    size_t bufferSize = mTensorSizeMap.at(name);

    // Create pinned buffer with reserved capacity
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

InferErrorCode TrtAlgoInference::setupCudaGraph() {
  // Not available in multi-threading mode. Temporarily disabled
  mCudaGraphEnabled = false;

  if (!mAllInputsStatic) {
    LOG_INFO_S << "CUDA Graph not available: model has dynamic input shapes.";
    return InferErrorCode::SUCCESS;
  }

  LOG_INFO_S
      << "CUDA Graph available but disabled by default for thread safety. "
      << "Model: " << mParams.name;

  return InferErrorCode::SUCCESS;
}

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

    err = setupCudaGraph();
    if (err != InferErrorCode::SUCCESS) {
      releaseResources();
      return err;
    }

    mIsInitialized = true;
    LOG_INFO_S << "TrtAlgoInference initialized successfully for model: "
               << mParams.name;
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during initialization: " << e.what();
    releaseResources();
    return InferErrorCode::INIT_FAILED;
  }
}

bool TrtAlgoInference::updateInputShapesIfNeeded(const TensorData &inputs) {
  bool shapeChanged = false;

  for (const auto &inputInfo : modelInfo->inputs) {
    const auto &name = inputInfo.name;
    const bool isDynamic = mDynamicInputTensorNames.count(name);

    if (!isDynamic) {
      continue; // Static shapes don't need updating
    }

    auto shapeIt = inputs.shapes.find(name);
    if (shapeIt == inputs.shapes.end()) {
      continue; // Will be caught by validation later
    }

    const std::vector<int64_t> newShape(shapeIt->second.begin(),
                                        shapeIt->second.end());
    auto cacheIt = mCachedInputShapes.find(name);

    // Check if shape changed
    if (cacheIt == mCachedInputShapes.end() || cacheIt->second != newShape) {
      // Shape changed, need to update
      nvinfer1::Dims actualDims;
      actualDims.nbDims = newShape.size();
      std::copy(newShape.begin(), newShape.end(), actualDims.d);

      if (!mContext->setInputShape(name.c_str(), actualDims)) {
        LOG_ERROR_S << "Failed to set input shape for tensor: " << name;
        return false;
      }

      // Update cache
      mCachedInputShapes[name] = newShape;
      shapeChanged = true;

      LOG_TRACE_S << "Updated input shape for tensor '" << name << "'";
    }
  }

  // If shapes changed and we had a captured graph, we need to re-capture
  if (shapeChanged && mCudaGraphCaptured) {
    LOG_DEBUG_S << "Input shapes changed, invalidating CUDA Graph.";
    if (mCudaGraphExec) {
      cudaGraphExecDestroy(mCudaGraphExec);
      mCudaGraphExec = nullptr;
    }
    if (mCudaGraph) {
      cudaGraphDestroy(mCudaGraph);
      mCudaGraph = nullptr;
    }
    mCudaGraphCaptured = false;
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
      LOG_TRACE_S << "Copying CPU input for tensor '" << name << "' (H2D).";
      const void *srcHostPtr = inputBuffer.getRawHostPtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(destDevicePtr, srcHostPtr,
                                       actualSizeBytes, cudaMemcpyHostToDevice,
                                       mStream));
    } else if (inputBuffer.location() == BufferLocation::GPU_DEVICE) {
      LOG_TRACE_S << "Copying GPU input for tensor '" << name << "' (D2D).";
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

  // Issue all D2H copies asynchronously
  for (const auto &outputInfo : modelInfo->outputs) {
    const auto &name = outputInfo.name;
    void *srcDevicePtr = mTensorAddressMap.at(name);

    nvinfer1::Dims actualOutputDims = mContext->getTensorShape(name.c_str());
    int64_t actualVolume = calculateVolume(actualOutputDims);

    size_t actualOutputSizeBytes =
        static_cast<size_t>(actualVolume) *
        trt_utils::getTrtElementSize(
            trt_utils::aiCoreDataTypeToTrt(outputInfo.dataType));

    // Use pre-allocated pinned buffer
    auto &pinnedBuffer = mPinnedOutputBuffers.at(name);

    if (pinnedBuffer.capacity() < actualOutputSizeBytes) {
      pinnedBuffer.reserve(actualOutputSizeBytes);
    }

    // Get write pointer (this sets size)
    uint8_t *destHostPtr = pinnedBuffer.writePtr(actualOutputSizeBytes);

    LOG_TRACE_S << "Copying output for tensor '" << name << "' to CPU (D2H).";
    CHECK_CUDA_ERROR(cudaMemcpyAsync(destHostPtr, srcDevicePtr,
                                     actualOutputSizeBytes,
                                     cudaMemcpyDeviceToHost, mStream));

    // Store shape info (CPU-side, safe to do now)
    outputs.shapes[name].assign(actualOutputDims.d,
                                actualOutputDims.d + actualOutputDims.nbDims);
  }

  // Wait for ALL D2H copies to complete
  CHECK_CUDA_ERROR(cudaStreamSynchronize(mStream));

  for (const auto &outputInfo : modelInfo->outputs) {
    const auto &name = outputInfo.name;
    auto &pinnedBuffer = mPinnedOutputBuffers.at(name);
    std::vector<uint8_t> safeData = pinnedBuffer.toVector();
    outputs.datas[name] =
        TypedBuffer::createFromCpu(outputInfo.dataType, std::move(safeData));
  }
}

InferErrorCode TrtAlgoInference::inferWithGraph(const TensorData &inputs,
                                                TensorData &outputs) {
  if (!mCudaGraphCaptured) {
    // Set capture flag to help diagnose issues if other CUDA ops fail
    mGraphCaptureInProgress.store(true, std::memory_order_release);

    LOG_INFO_S << "Capturing CUDA Graph for model: " << mParams.name;

    // Copy inputs first (this is NOT part of the graph - inputs vary each call)
    copyInputsToDevice(inputs);

    // Synchronize to ensure inputs are ready before capture
    CHECK_CUDA_ERROR(cudaStreamSynchronize(mStream));

    // Start graph capture with ThreadLocal mode to avoid blocking other threads
    // WARNING: Even with ThreadLocal mode, operations like
    // cudaDeviceSynchronize() are still forbidden globally during capture. This
    // can cause issues if other threads are running GPU preprocessing
    // concurrently.
    cudaError_t captureErr =
        cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal);
    if (captureErr != cudaSuccess) {
      mGraphCaptureInProgress.store(false, std::memory_order_release);
      LOG_WARNING_S << "Failed to begin CUDA Graph capture: "
                    << cudaGetErrorString(captureErr)
                    << ". Falling back to non-graph inference.";
      mCudaGraphEnabled = false;
      return inferWithoutGraph(inputs, outputs);
    }

    // Execute inference (this will be captured, but NOT executed during
    // capture!)
    bool enqueueSuccess = mContext->enqueueV3(mStream);

    // End capture - must be called even if enqueue failed
    cudaGraph_t tempGraph = nullptr;
    captureErr = cudaStreamEndCapture(mStream, &tempGraph);

    // Clear capture flag AFTER EndCapture
    mGraphCaptureInProgress.store(false, std::memory_order_release);

    if (!enqueueSuccess || captureErr != cudaSuccess || tempGraph == nullptr) {
      LOG_WARNING_S
          << "CUDA Graph capture failed. Falling back to non-graph inference.";
      if (tempGraph) {
        cudaGraphDestroy(tempGraph);
      }
      mCudaGraphEnabled = false;
      // Need to re-run inference since capture consumed it
      return inferWithoutGraph(inputs, outputs);
    }

    mCudaGraph = tempGraph;

    // Instantiate the graph for execution
    cudaGraphExec_t tempGraphExec = nullptr;
    captureErr =
        cudaGraphInstantiate(&tempGraphExec, mCudaGraph, nullptr, nullptr, 0);
    if (captureErr != cudaSuccess || tempGraphExec == nullptr) {
      LOG_WARNING_S << "Failed to instantiate CUDA Graph: "
                    << cudaGetErrorString(captureErr)
                    << ". Falling back to non-graph inference.";
      cudaGraphDestroy(mCudaGraph);
      mCudaGraph = nullptr;
      mCudaGraphEnabled = false;
      return inferWithoutGraph(inputs, outputs);
    }

    mCudaGraphExec = tempGraphExec;
    mCudaGraphCaptured = true;
    LOG_INFO_S << "CUDA Graph captured successfully.";

    CHECK_CUDA_ERROR(cudaGraphLaunch(mCudaGraphExec, mStream));

    copyOutputsToHost(outputs);

    return InferErrorCode::SUCCESS;
  }

  // Subsequent inferences: copy inputs and launch graph
  copyInputsToDevice(inputs);

  // Launch the captured graph (this is much faster than enqueueV3)
  CHECK_CUDA_ERROR(cudaGraphLaunch(mCudaGraphExec, mStream));

  // Copy outputs (includes internal synchronization)
  copyOutputsToHost(outputs);

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::inferWithoutGraph(const TensorData &inputs,
                                                   TensorData &outputs) {
  // Standard inference path without CUDA Graph
  copyInputsToDevice(inputs);

  LOG_TRACE_S << "Executing inference on stream " << mStream;
  if (!mContext->enqueueV3(mStream)) {
    LOG_ERROR_S << "Failed to enqueue TensorRT inference.";
    return InferErrorCode::INFER_EXECUTION_FAILED;
  }

  copyOutputsToHost(outputs);

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::infer(const TensorData &inputs,
                                       TensorData &outputs) {
  std::lock_guard<std::mutex> lock(mMutex);

  if (!mIsInitialized) {
    LOG_ERROR_S << "Inference called on uninitialized model.";
    return InferErrorCode::NOT_INITIALIZED;
  }

  try {
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

    InferErrorCode result;
    if (mCudaGraphEnabled && mAllInputsStatic) {
      result = inferWithGraph(inputs, outputs);
    } else {
      result = inferWithoutGraph(inputs, outputs);
    }

    LOG_TRACE_S << "Inference completed successfully.";
    return result;

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

int64_t TrtAlgoInference::calculateVolume(const nvinfer1::Dims &dims) {
  return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1},
                         std::multiplies<int64_t>());
}

} // namespace ai_core::dnn
