/**
 * @file trt_infer_stream.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief TensorRT inference stream implementation
 * @version 0.1
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "trt_infer_stream.hpp"
#include "ai_core/logger.hpp"
#include "trt_infer.hpp"
#include "trt_utils.hpp"
#include <cstring>
#include <future>
#include <numeric>

namespace ai_core::dnn {

// ============================================================================
// Constructor / Destructor
// ============================================================================

TrtInferStream::TrtInferStream(TrtAlgoInference &engine)
    : mEngine(engine), mSharedEngine(engine.mEngine), mCudaStream(nullptr),
      mContext(nullptr) {}

TrtInferStream::~TrtInferStream() {
  destroyGraph();

  mContext.reset();

  if (mCudaStream) {
    cudaError_t err = cudaStreamDestroy(mCudaStream);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG_WARNING_S << "Failed to destroy CUDA stream: "
                    << cudaGetErrorString(err);
    }
    mCudaStream = nullptr;
  }

  mDeviceBuffers.clear();
  mPinnedOutputBuffers.clear();
  mTensorAddressMap.clear();
  mCachedInputShapes.clear();

  LOG_DEBUG_S << "TrtInferStream destroyed";
}

// ============================================================================
// IExecutionContext Implementation
// ============================================================================

std::future<InferErrorCode> TrtInferStream::inferAsync(const TensorData &inputs,
                                                       TensorData &outputs) {
  // Early validation - return immediately settled future on error
  if (!mInitialized) {
    std::promise<InferErrorCode> promise;
    promise.set_value(InferErrorCode::NOT_INITIALIZED);
    return promise.get_future();
  }

  try {
    // Update input shapes if dynamic
    if (!updateInputShapesIfNeeded(inputs)) {
      std::promise<InferErrorCode> promise;
      promise.set_value(InferErrorCode::INFER_EXECUTION_FAILED);
      return promise.get_future();
    }

    // Submit all async operations (non-blocking)

    // H2D: Copy inputs to device (async)
    auto copyResult = copyInputsToDevice(inputs);
    if (copyResult != InferErrorCode::SUCCESS) {
      std::promise<InferErrorCode> promise;
      promise.set_value(copyResult);
      return promise.get_future();
    }

    // Execute: Run inference kernel (async)
    InferErrorCode execResult;
    if (mGraphEnabled && mGraphCaptured) {
      execResult = launchGraph();
    } else if (mGraphEnabled && !mGraphCaptured) {
      execResult = captureGraph();
    } else {
      execResult = executeInference();
    }

    if (execResult != InferErrorCode::SUCCESS) {
      std::promise<InferErrorCode> promise;
      promise.set_value(execResult);
      return promise.get_future();
    }

    // D2H: Submit async output copy (non-blocking)
    auto d2hResult = submitAsyncD2H(outputs);
    if (d2hResult != InferErrorCode::SUCCESS) {
      std::promise<InferErrorCode> promise;
      promise.set_value(d2hResult);
      return promise.get_future();
    }

    // Return deferred future for caller-side synchronization
    //
    // Key design: No thread is created here!
    // The lambda executes in the caller's thread when future.get() is called.
    // This eliminates thread creation overhead (~100-500μs per call).

    cudaStream_t stream = mCudaStream;

    return std::async(std::launch::deferred,
                      [this, stream, &outputs]() -> InferErrorCode {
                        // Synchronize CUDA stream (blocks until all ops
                        // complete)
                        cudaError_t err = cudaStreamSynchronize(stream);
                        if (err != cudaSuccess) {
                          LOG_ERROR_S << "CUDA stream synchronize failed: "
                                      << cudaGetErrorString(err);
                          return InferErrorCode::STREAM_SYNC_FAILED;
                        }

                        // Finalize: Copy from pinned buffers to output
                        return finalizeOutputs(outputs);
                      });

  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception in inferAsync: " << e.what();
    std::promise<InferErrorCode> promise;
    promise.set_value(InferErrorCode::INFER_FAILED);
    return promise.get_future();
  }
}

InferErrorCode TrtInferStream::synchronize() {
  if (!mInitialized) {
    return InferErrorCode::NOT_INITIALIZED;
  }

  cudaError_t err = cudaStreamSynchronize(mCudaStream);
  if (err != cudaSuccess) {
    LOG_ERROR_S << "CUDA stream synchronize failed: "
                << cudaGetErrorString(err);
    return InferErrorCode::STREAM_SYNC_FAILED;
  }
  return InferErrorCode::SUCCESS;
}

bool TrtInferStream::isComplete() const {
  if (!mInitialized || !mCudaStream) {
    return true;
  }
  cudaError_t status = cudaStreamQuery(mCudaStream);
  return status == cudaSuccess;
}

BackendHandle TrtInferStream::getHandle() const {
  return BackendHandle(mCudaStream);
}

InferErrorCode TrtInferStream::setGraphEnabled(bool enable) {
  if (!mInitialized) {
    return InferErrorCode::NOT_INITIALIZED;
  }

  if (enable && !mEngine.mAllInputsStatic) {
    LOG_WARNING_S << "CUDA Graph requires static input shapes. "
                  << "Graph will remain disabled.";
    return InferErrorCode::SUCCESS;
  }

  if (enable == mGraphEnabled) {
    return InferErrorCode::SUCCESS;
  }

  if (!enable && mGraphCaptured) {
    destroyGraph();
  }

  mGraphEnabled = enable;
  mGraphCaptured = false;

  LOG_INFO_S << "CUDA Graph " << (enable ? "enabled" : "disabled")
             << " for stream";
  return InferErrorCode::SUCCESS;
}

bool TrtInferStream::isGraphEnabled() const { return mGraphEnabled; }

// ============================================================================
// Stream Lifecycle
// ============================================================================

InferErrorCode TrtInferStream::initialize() {
  if (mInitialized) {
    return InferErrorCode::SUCCESS;
  }

  if (!mSharedEngine) {
    LOG_ERROR_S << "Parent engine not initialized";
    return InferErrorCode::NOT_INITIALIZED;
  }

  // Create high-priority non-blocking CUDA stream
  int leastPriority, greatestPriority;
  CHECK_CUDA_ERROR(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(
      &mCudaStream, cudaStreamNonBlocking, greatestPriority));

  // Create execution context
  mContext.reset(mSharedEngine->createExecutionContext());
  if (!mContext) {
    LOG_ERROR_S << "Failed to create TensorRT execution context for stream";
    return InferErrorCode::INIT_CONTEXT_FAILED;
  }

  // Allocate device buffers
  auto allocResult = allocateBuffers();
  if (allocResult != InferErrorCode::SUCCESS) {
    return allocResult;
  }

  // Allocate pinned output buffers
  auto pinnedResult = allocatePinnedOutputBuffers();
  if (pinnedResult != InferErrorCode::SUCCESS) {
    return pinnedResult;
  }

  mInitialized = true;
  LOG_DEBUG_S << "TrtInferStream initialized successfully";
  return InferErrorCode::SUCCESS;
}

// ============================================================================
// Internal Methods
// ============================================================================

InferErrorCode TrtInferStream::allocateBuffers() {
  const int32_t numIOTensors = mSharedEngine->getNbIOTensors();
  const int profileIndex = 0;

  mDeviceBuffers.clear();
  mDeviceBuffers.reserve(numIOTensors);
  mTensorAddressMap.clear();

  // Set input shapes to MAX for buffer allocation
  for (int32_t i = 0; i < numIOTensors; ++i) {
    const char *name = mSharedEngine->getIOTensorName(i);
    if (mSharedEngine->getTensorIOMode(name) ==
        nvinfer1::TensorIOMode::kINPUT) {
      auto maxDims = mSharedEngine->getProfileShape(
          name, profileIndex, nvinfer1::OptProfileSelector::kMAX);
      if (!mContext->setInputShape(name, maxDims)) {
        LOG_WARNING_S << "Failed to set max input shape for: " << name;
      }
    }
  }

  // Allocate buffers for each tensor
  for (int32_t i = 0; i < numIOTensors; ++i) {
    const char *name = mSharedEngine->getIOTensorName(i);
    size_t bufferSize = mEngine.mTensorSizeMap.at(name);

    mDeviceBuffers.emplace_back(cuda_utils::DeviceByteBuffer{bufferSize});
    void *devicePtr = mDeviceBuffers.back().unsafePtr();

    mTensorAddressMap[name] = devicePtr;

    if (!mContext->setTensorAddress(name, devicePtr)) {
      LOG_ERROR_S << "Failed to set tensor address for: " << name;
      return InferErrorCode::INIT_BINDING_FAILED;
    }

    LOG_DEBUG_S << "Stream allocated " << bufferSize
                << " bytes for tensor: " << name;
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::allocatePinnedOutputBuffers() {
  for (const auto &outputInfo : mEngine.modelInfo->outputs) {
    const auto &name = outputInfo.name;
    size_t bufferSize = mEngine.mTensorSizeMap.at(name);

    cuda_utils::CudaHostBuffer<uint8_t> pinnedBuffer;
    pinnedBuffer.reserve(bufferSize);

    mPinnedOutputBuffers[name] = std::move(pinnedBuffer);
  }

  return InferErrorCode::SUCCESS;
}

bool TrtInferStream::updateInputShapesIfNeeded(const TensorData &inputs) {
  bool shapeChanged = false;

  for (const auto &inputInfo : mEngine.modelInfo->inputs) {
    const auto &name = inputInfo.name;
    const bool isDynamic = mEngine.mDynamicInputTensorNames.count(name);

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
      shapeChanged = true;
    }
  }

  // If shapes changed and we had a captured graph, invalidate it
  if (shapeChanged && mGraphCaptured) {
    LOG_DEBUG_S << "Input shapes changed, invalidating CUDA Graph.";
    destroyGraph();
    mGraphCaptured = false;
  }

  return true;
}

InferErrorCode TrtInferStream::copyInputsToDevice(const TensorData &inputs) {
  for (const auto &inputInfo : mEngine.modelInfo->inputs) {
    const auto &name = inputInfo.name;

    auto dataIt = inputs.datas.find(name);
    if (dataIt == inputs.datas.end()) {
      LOG_ERROR_S << "Input tensor not found: " << name;
      return InferErrorCode::INFER_INPUT_ERROR;
    }

    const TypedBuffer &inputBuffer = dataIt->second;
    const size_t actualSizeBytes = inputBuffer.getSizeBytes();
    void *destDevicePtr = mTensorAddressMap.at(name);

    if (inputBuffer.location() == BufferLocation::CPU) {
      const void *srcHostPtr = inputBuffer.getRawHostPtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(destDevicePtr, srcHostPtr,
                                       actualSizeBytes, cudaMemcpyHostToDevice,
                                       mCudaStream));
    } else if (inputBuffer.location() == BufferLocation::GPU_DEVICE) {
      void *srcDevicePtr = inputBuffer.getRawDevicePtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(destDevicePtr, srcDevicePtr,
                                       actualSizeBytes,
                                       cudaMemcpyDeviceToDevice, mCudaStream));
    }
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::submitAsyncD2H(TensorData &outputs) {
  outputs.datas.clear();
  outputs.shapes.clear();

  for (const auto &outputInfo : mEngine.modelInfo->outputs) {
    const auto &name = outputInfo.name;
    void *srcDevicePtr = mTensorAddressMap.at(name);

    nvinfer1::Dims actualOutputDims = mContext->getTensorShape(name.c_str());
    int64_t actualVolume = TrtAlgoInference::calculateVolume(actualOutputDims);

    size_t actualOutputSizeBytes =
        static_cast<size_t>(actualVolume) *
        trt_utils::getTrtElementSize(
            trt_utils::aiCoreDataTypeToTrt(outputInfo.dataType));

    auto &pinnedBuffer = mPinnedOutputBuffers.at(name);

    if (pinnedBuffer.capacity() < actualOutputSizeBytes) {
      pinnedBuffer.reserve(actualOutputSizeBytes);
    }

    uint8_t *destHostPtr = pinnedBuffer.writePtr(actualOutputSizeBytes);

    // Submit async D2H copy (non-blocking)
    CHECK_CUDA_ERROR(cudaMemcpyAsync(destHostPtr, srcDevicePtr,
                                     actualOutputSizeBytes,
                                     cudaMemcpyDeviceToHost, mCudaStream));

    // Store output shapes immediately (available from context)
    outputs.shapes[name].assign(actualOutputDims.d,
                                actualOutputDims.d + actualOutputDims.nbDims);
  }

  // NOTE: No synchronization here! Caller is responsible for sync.
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::finalizeOutputs(TensorData &outputs) {
  // Copy from pinned buffers to output TypedBuffers
  // PRECONDITION: cudaStreamSynchronize must have been called!
  for (const auto &outputInfo : mEngine.modelInfo->outputs) {
    const auto &name = outputInfo.name;
    auto &pinnedBuffer = mPinnedOutputBuffers.at(name);
    std::vector<uint8_t> safeData = pinnedBuffer.toVector();
    outputs.datas[name] =
        TypedBuffer::createFromCpu(outputInfo.dataType, std::move(safeData));
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::copyOutputsToHost(TensorData &outputs) {
  // Legacy synchronous version for backward compatibility
  auto result = submitAsyncD2H(outputs);
  if (result != InferErrorCode::SUCCESS) {
    return result;
  }

  // Synchronize to ensure D2H copies complete
  CHECK_CUDA_ERROR(cudaStreamSynchronize(mCudaStream));

  return finalizeOutputs(outputs);
}

InferErrorCode TrtInferStream::executeInference() {
  if (!mContext->enqueueV3(mCudaStream)) {
    LOG_ERROR_S << "TensorRT enqueueV3 failed";
    return InferErrorCode::INFER_EXECUTION_FAILED;
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::captureGraph() {
  LOG_INFO_S << "Capturing CUDA Graph...";

  // Warm-up run before capture
  if (!mContext->enqueueV3(mCudaStream)) {
    LOG_ERROR_S << "Warm-up enqueue failed before graph capture";
    return InferErrorCode::INFER_EXECUTION_FAILED;
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(mCudaStream));

  // Begin capture with ThreadLocal mode
  cudaError_t captureErr =
      cudaStreamBeginCapture(mCudaStream, cudaStreamCaptureModeThreadLocal);
  if (captureErr != cudaSuccess) {
    LOG_WARNING_S << "Failed to begin CUDA Graph capture: "
                  << cudaGetErrorString(captureErr)
                  << ". Falling back to non-graph inference.";
    mGraphEnabled = false;
    return executeInference();
  }

  // Execute inference (will be captured)
  bool enqueueSuccess = mContext->enqueueV3(mCudaStream);

  // End capture
  cudaGraph_t tempGraph = nullptr;
  captureErr = cudaStreamEndCapture(mCudaStream, &tempGraph);

  if (!enqueueSuccess || captureErr != cudaSuccess || tempGraph == nullptr) {
    LOG_WARNING_S << "CUDA Graph capture failed. Falling back to non-graph.";
    if (tempGraph) {
      cudaGraphDestroy(tempGraph);
    }
    mGraphEnabled = false;
    return executeInference();
  }

  mCudaGraph = tempGraph;

  // Instantiate the graph
  cudaGraphExec_t tempGraphExec = nullptr;
  captureErr =
      cudaGraphInstantiate(&tempGraphExec, mCudaGraph, nullptr, nullptr, 0);
  if (captureErr != cudaSuccess || tempGraphExec == nullptr) {
    LOG_WARNING_S << "Failed to instantiate CUDA Graph: "
                  << cudaGetErrorString(captureErr);
    cudaGraphDestroy(mCudaGraph);
    mCudaGraph = nullptr;
    mGraphEnabled = false;
    return executeInference();
  }

  mCudaGraphExec = tempGraphExec;
  mGraphCaptured = true;
  LOG_INFO_S << "CUDA Graph captured successfully.";

  // Launch the graph for this first inference
  CHECK_CUDA_ERROR(cudaGraphLaunch(mCudaGraphExec, mCudaStream));

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::launchGraph() {
  if (!mCudaGraphExec) {
    LOG_ERROR_S << "No graph to launch";
    return InferErrorCode::GRAPH_LAUNCH_FAILED;
  }

  cudaError_t err = cudaGraphLaunch(mCudaGraphExec, mCudaStream);
  if (err != cudaSuccess) {
    LOG_ERROR_S << "cudaGraphLaunch failed: " << cudaGetErrorString(err);
    return InferErrorCode::GRAPH_LAUNCH_FAILED;
  }

  return InferErrorCode::SUCCESS;
}

void TrtInferStream::destroyGraph() {
  if (mCudaGraphExec) {
    cudaGraphExecDestroy(mCudaGraphExec);
    mCudaGraphExec = nullptr;
  }
  if (mCudaGraph) {
    cudaGraphDestroy(mCudaGraph);
    mCudaGraph = nullptr;
  }
  mGraphCaptured = false;
}

} // namespace ai_core::dnn