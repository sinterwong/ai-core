/**
 * @file trt_infer_stream.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief TensorRT inference stream implementation
 * @version 0.1
 * @date 2026-01-06
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
    : m_engine(engine), m_sharedEngine(engine.m_engine), m_cudaStream(nullptr),
      m_context(nullptr) {}

TrtInferStream::~TrtInferStream() {
  destroyGraph();

  m_context.reset();

  if (m_cudaStream) {
    cudaError_t err = cudaStreamDestroy(m_cudaStream);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG_WARNING_S << "Failed to destroy CUDA stream: "
                    << cudaGetErrorString(err);
    }
    m_cudaStream = nullptr;
  }

  m_deviceBuffers.clear();
  m_pinnedOutputBuffers.clear();
  m_tensorAddressMap.clear();
  m_cachedInputShapes.clear();

  LOG_DEBUG_S << "TrtInferStream destroyed";
}

// ============================================================================
// IExecutionContext Implementation
// ============================================================================

std::future<InferErrorCode> TrtInferStream::inferAsync(const TensorData &inputs,
                                                       TensorData &outputs) {
  // Early validation - return immediately settled future on error
  if (!m_initialized) {
    std::promise<InferErrorCode> promise;
    promise.set_value(InferErrorCode::NotInitialized);
    return promise.get_future();
  }

  try {
    // Update input shapes if dynamic
    if (!updateInputShapesIfNeeded(inputs)) {
      std::promise<InferErrorCode> promise;
      promise.set_value(InferErrorCode::InferExecutionFailed);
      return promise.get_future();
    }

    // Submit all async operations (non-blocking)

    // H2D: Copy inputs to device (async)
    auto copy_result = copyInputsToDevice(inputs);
    if (copy_result != InferErrorCode::SUCCESS) {
      std::promise<InferErrorCode> promise;
      promise.set_value(copy_result);
      return promise.get_future();
    }

    // Execute: Run inference kernel (async)
    InferErrorCode exec_result;
    if (m_graphEnabled && m_graphCaptured) {
      exec_result = launchGraph();
    } else if (m_graphEnabled && !m_graphCaptured) {
      exec_result = captureGraph();
    } else {
      exec_result = executeInference();
    }

    if (exec_result != InferErrorCode::SUCCESS) {
      std::promise<InferErrorCode> promise;
      promise.set_value(exec_result);
      return promise.get_future();
    }

    // D2H: Submit async output copy (non-blocking)
    auto d2h_result = submitAsyncD2H(outputs);
    if (d2h_result != InferErrorCode::SUCCESS) {
      std::promise<InferErrorCode> promise;
      promise.set_value(d2h_result);
      return promise.get_future();
    }

    // Return deferred future for caller-side synchronization
    //
    // Key design: No thread is created here!
    // The lambda executes in the caller's thread when future.get() is called.
    // This eliminates thread creation overhead (~100-500μs per call).

    cudaStream_t stream = m_cudaStream;

    return std::async(std::launch::deferred,
                      [this, stream, &outputs]() -> InferErrorCode {
                        // Synchronize CUDA stream (blocks until all ops
                        // complete)
                        cudaError_t err = cudaStreamSynchronize(stream);
                        if (err != cudaSuccess) {
                          LOG_ERROR_S << "CUDA stream synchronize failed: "
                                      << cudaGetErrorString(err);
                          return InferErrorCode::StreamSyncFailed;
                        }

                        // Finalize: Copy from pinned buffers to output
                        return finalizeOutputs(outputs);
                      });

  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception in inferAsync: " << e.what();
    std::promise<InferErrorCode> promise;
    promise.set_value(InferErrorCode::InferFailed);
    return promise.get_future();
  }
}

InferErrorCode TrtInferStream::synchronize() {
  if (!m_initialized) {
    return InferErrorCode::NotInitialized;
  }

  cudaError_t err = cudaStreamSynchronize(m_cudaStream);
  if (err != cudaSuccess) {
    LOG_ERROR_S << "CUDA stream synchronize failed: "
                << cudaGetErrorString(err);
    return InferErrorCode::StreamSyncFailed;
  }
  return InferErrorCode::SUCCESS;
}

bool TrtInferStream::isComplete() const {
  if (!m_initialized || !m_cudaStream) {
    return true;
  }
  cudaError_t status = cudaStreamQuery(m_cudaStream);
  return status == cudaSuccess;
}

BackendHandle TrtInferStream::getHandle() const {
  return BackendHandle(m_cudaStream);
}

InferErrorCode TrtInferStream::setGraphEnabled(bool enable) {
  if (!m_initialized) {
    return InferErrorCode::NotInitialized;
  }

  if (enable && !m_engine.m_allInputsStatic) {
    LOG_WARNING_S << "CUDA Graph requires static input shapes. "
                  << "Graph will remain disabled.";
    return InferErrorCode::SUCCESS;
  }

  if (enable == m_graphEnabled) {
    return InferErrorCode::SUCCESS;
  }

  if (!enable && m_graphCaptured) {
    destroyGraph();
  }

  m_graphEnabled = enable;
  m_graphCaptured = false;

  LOG_INFO_S << "CUDA Graph " << (enable ? "enabled" : "disabled")
             << " for stream";
  return InferErrorCode::SUCCESS;
}

bool TrtInferStream::isGraphEnabled() const { return m_graphEnabled; }

// ============================================================================
// Stream Lifecycle
// ============================================================================

InferErrorCode TrtInferStream::initialize() {
  if (m_initialized) {
    return InferErrorCode::SUCCESS;
  }

  if (!m_sharedEngine) {
    LOG_ERROR_S << "Parent engine not initialized";
    return InferErrorCode::NotInitialized;
  }

  // Create high-priority non-blocking CUDA stream
  int least_priority, greatest_priority;
  CHECK_CUDA_ERROR(
      cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(
      &m_cudaStream, cudaStreamNonBlocking, greatest_priority));

  // Create execution context
  m_context.reset(m_sharedEngine->createExecutionContext());
  if (!m_context) {
    LOG_ERROR_S << "Failed to create TensorRT execution context for stream";
    return InferErrorCode::InitContextFailed;
  }

  // Allocate device buffers
  auto alloc_result = allocateBuffers();
  if (alloc_result != InferErrorCode::SUCCESS) {
    return alloc_result;
  }

  // Allocate pinned output buffers
  auto pinned_result = allocatePinnedOutputBuffers();
  if (pinned_result != InferErrorCode::SUCCESS) {
    return pinned_result;
  }

  m_initialized = true;
  LOG_DEBUG_S << "TrtInferStream initialized successfully";
  return InferErrorCode::SUCCESS;
}

// ============================================================================
// Internal Methods
// ============================================================================

InferErrorCode TrtInferStream::allocateBuffers() {
  const int32_t num_io_tensors = m_sharedEngine->getNbIOTensors();
  const int profile_index = 0;

  m_deviceBuffers.clear();
  m_deviceBuffers.reserve(num_io_tensors);
  m_tensorAddressMap.clear();

  // Set input shapes to MAX for buffer allocation
  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char *name = m_sharedEngine->getIOTensorName(i);
    if (m_sharedEngine->getTensorIOMode(name) ==
        nvinfer1::TensorIOMode::kINPUT) {
      auto max_dims = m_sharedEngine->getProfileShape(
          name, profile_index, nvinfer1::OptProfileSelector::kMAX);
      if (!m_context->setInputShape(name, max_dims)) {
        LOG_WARNING_S << "Failed to set max input shape for: " << name;
      }
    }
  }

  // Allocate buffers for each tensor
  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char *name = m_sharedEngine->getIOTensorName(i);
    size_t buffer_size = m_engine.m_tensorSizeMap.at(name);

    m_deviceBuffers.emplace_back(cuda_utils::DeviceByteBuffer{buffer_size});
    void *device_ptr = m_deviceBuffers.back().unsafePtr();

    m_tensorAddressMap[name] = device_ptr;

    if (!m_context->setTensorAddress(name, device_ptr)) {
      LOG_ERROR_S << "Failed to set tensor address for: " << name;
      return InferErrorCode::InitBindingFailed;
    }

    LOG_DEBUG_S << "Stream allocated " << buffer_size
                << " bytes for tensor: " << name;
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::allocatePinnedOutputBuffers() {
  for (const auto &output_info : m_engine.m_modelInfo->outputs) {
    const auto &name = output_info.name;
    size_t buffer_size = m_engine.m_tensorSizeMap.at(name);

    cuda_utils::CudaHostBuffer<uint8_t> pinned_buffer;
    pinned_buffer.reserve(buffer_size);

    m_pinnedOutputBuffers[name] = std::move(pinned_buffer);
  }

  return InferErrorCode::SUCCESS;
}

bool TrtInferStream::updateInputShapesIfNeeded(const TensorData &inputs) {
  bool shape_changed = false;

  for (const auto &input_info : m_engine.m_modelInfo->inputs) {
    const auto &name = input_info.name;
    const bool is_dynamic = m_engine.m_dynamicInputTensorNames.count(name);

    if (!is_dynamic) {
      continue;
    }

    const Tensor *input_tensor = inputs.find(name);
    if (input_tensor == nullptr || input_tensor->shape.empty()) {
      continue;
    }

    const std::vector<int64_t> new_shape(input_tensor->shape.begin(),
                                         input_tensor->shape.end());
    auto cache_it = m_cachedInputShapes.find(name);

    if (cache_it == m_cachedInputShapes.end() ||
        cache_it->second != new_shape) {
      nvinfer1::Dims actual_dims;
      actual_dims.nbDims = new_shape.size();
      std::copy(new_shape.begin(), new_shape.end(), actual_dims.d);

      if (!m_context->setInputShape(name.c_str(), actual_dims)) {
        LOG_ERROR_S << "Failed to set input shape for tensor: " << name;
        return false;
      }

      m_cachedInputShapes[name] = new_shape;
      shape_changed = true;
    }
  }

  // If shapes changed and we had a captured graph, invalidate it
  if (shape_changed && m_graphCaptured) {
    LOG_DEBUG_S << "Input shapes changed, invalidating CUDA Graph.";
    destroyGraph();
    m_graphCaptured = false;
  }

  return true;
}

InferErrorCode TrtInferStream::copyInputsToDevice(const TensorData &inputs) {
  for (const auto &input_info : m_engine.m_modelInfo->inputs) {
    const auto &name = input_info.name;

    const Tensor *input_tensor = inputs.find(name);
    if (input_tensor == nullptr) {
      LOG_ERROR_S << "Input tensor not found: " << name;
      return InferErrorCode::InferInputError;
    }

    const TypedBuffer &input_buffer = input_tensor->buffer;
    const size_t actual_size_bytes = input_buffer.getSizeBytes();
    void *dest_device_ptr = m_tensorAddressMap.at(name);

    if (input_buffer.location() == BufferLocation::CPU) {
      const void *src_host_ptr = input_buffer.getRawHostPtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dest_device_ptr, src_host_ptr,
                                       actual_size_bytes,
                                       cudaMemcpyHostToDevice, m_cudaStream));
    } else if (input_buffer.location() == BufferLocation::GpuDevice) {
      void *src_device_ptr = input_buffer.getRawDevicePtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dest_device_ptr, src_device_ptr,
                                       actual_size_bytes,
                                       cudaMemcpyDeviceToDevice, m_cudaStream));
    }
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::submitAsyncD2H(TensorData &outputs) {
  outputs.clear();

  for (const auto &output_info : m_engine.m_modelInfo->outputs) {
    const auto &name = output_info.name;
    void *src_device_ptr = m_tensorAddressMap.at(name);

    nvinfer1::Dims actual_output_dims = m_context->getTensorShape(name.c_str());
    int64_t actual_volume =
        TrtAlgoInference::calculateVolume(actual_output_dims);

    size_t actual_output_size_bytes =
        static_cast<size_t>(actual_volume) *
        trt_utils::getTrtElementSize(
            trt_utils::aiCoreDataTypeToTrt(output_info.data_type));

    auto &pinned_buffer = m_pinnedOutputBuffers.at(name);

    if (pinned_buffer.capacity() < actual_output_size_bytes) {
      pinned_buffer.reserve(actual_output_size_bytes);
    }

    uint8_t *dest_host_ptr = pinned_buffer.writePtr(actual_output_size_bytes);

    // Submit async D2H copy (non-blocking)
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dest_host_ptr, src_device_ptr,
                                     actual_output_size_bytes,
                                     cudaMemcpyDeviceToHost, m_cudaStream));

    // Store output shapes immediately (available from context); the buffer
    // is filled in by finalizeOutputs after stream sync.
    outputs.set(
        name, TypedBuffer(),
        std::vector<int>(actual_output_dims.d,
                         actual_output_dims.d + actual_output_dims.nbDims));
  }

  // NOTE: No synchronization here! Caller is responsible for sync.
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::finalizeOutputs(TensorData &outputs) {
  // Copy from pinned buffers to output TypedBuffers
  // PRECONDITION: cudaStreamSynchronize must have been called!
  for (const auto &output_info : m_engine.m_modelInfo->outputs) {
    const auto &name = output_info.name;
    auto &pinned_buffer = m_pinnedOutputBuffers.at(name);
    std::vector<uint8_t> safe_data = pinned_buffer.toVector();
    Tensor *out = outputs.find(name);
    if (out == nullptr) {
      LOG_ERROR_S << "finalizeOutputs called before submitAsyncD2H for: "
                  << name;
      return InferErrorCode::InferOutputError;
    }
    out->buffer =
        TypedBuffer::createFromCpu(output_info.data_type, std::move(safe_data));
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
  CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cudaStream));

  return finalizeOutputs(outputs);
}

InferErrorCode TrtInferStream::executeInference() {
  if (!m_context->enqueueV3(m_cudaStream)) {
    LOG_ERROR_S << "TensorRT enqueueV3 failed";
    return InferErrorCode::InferExecutionFailed;
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::captureGraph() {
  LOG_INFO_S << "Capturing CUDA Graph...";

  // Warm-up run before capture
  if (!m_context->enqueueV3(m_cudaStream)) {
    LOG_ERROR_S << "Warm-up enqueue failed before graph capture";
    return InferErrorCode::InferExecutionFailed;
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cudaStream));

  // Begin capture with ThreadLocal mode
  cudaError_t capture_err =
      cudaStreamBeginCapture(m_cudaStream, cudaStreamCaptureModeThreadLocal);
  if (capture_err != cudaSuccess) {
    LOG_WARNING_S << "Failed to begin CUDA Graph capture: "
                  << cudaGetErrorString(capture_err)
                  << ". Falling back to non-graph inference.";
    m_graphEnabled = false;
    return executeInference();
  }

  // Execute inference (will be captured)
  bool enqueue_success = m_context->enqueueV3(m_cudaStream);

  // End capture
  cudaGraph_t temp_graph = nullptr;
  capture_err = cudaStreamEndCapture(m_cudaStream, &temp_graph);

  if (!enqueue_success || capture_err != cudaSuccess || temp_graph == nullptr) {
    LOG_WARNING_S << "CUDA Graph capture failed. Falling back to non-graph.";
    if (temp_graph) {
      cudaGraphDestroy(temp_graph);
    }
    m_graphEnabled = false;
    return executeInference();
  }

  m_cudaGraph = temp_graph;

  // Instantiate the graph
  cudaGraphExec_t temp_graph_exec = nullptr;
  capture_err =
      cudaGraphInstantiate(&temp_graph_exec, m_cudaGraph, nullptr, nullptr, 0);
  if (capture_err != cudaSuccess || temp_graph_exec == nullptr) {
    LOG_WARNING_S << "Failed to instantiate CUDA Graph: "
                  << cudaGetErrorString(capture_err);
    cudaGraphDestroy(m_cudaGraph);
    m_cudaGraph = nullptr;
    m_graphEnabled = false;
    return executeInference();
  }

  m_cudaGraphExec = temp_graph_exec;
  m_graphCaptured = true;
  LOG_INFO_S << "CUDA Graph captured successfully.";

  // Launch the graph for this first inference
  CHECK_CUDA_ERROR(cudaGraphLaunch(m_cudaGraphExec, m_cudaStream));

  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtInferStream::launchGraph() {
  if (!m_cudaGraphExec) {
    LOG_ERROR_S << "No graph to launch";
    return InferErrorCode::GraphLaunchFailed;
  }

  cudaError_t err = cudaGraphLaunch(m_cudaGraphExec, m_cudaStream);
  if (err != cudaSuccess) {
    LOG_ERROR_S << "cudaGraphLaunch failed: " << cudaGetErrorString(err);
    return InferErrorCode::GraphLaunchFailed;
  }

  return InferErrorCode::SUCCESS;
}

void TrtInferStream::destroyGraph() {
  if (m_cudaGraphExec) {
    cudaGraphExecDestroy(m_cudaGraphExec);
    m_cudaGraphExec = nullptr;
  }
  if (m_cudaGraph) {
    cudaGraphDestroy(m_cudaGraph);
    m_cudaGraph = nullptr;
  }
  m_graphCaptured = false;
}

} // namespace ai_core::dnn