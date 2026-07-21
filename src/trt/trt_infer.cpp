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
#include "ai_core/error_code.hpp"
#include "crypto.hpp"
#include "trt_infer_stream.hpp"
#include "trt_utils.hpp"
#include <filesystem>
#include <fstream>
#include <numeric>

namespace ai_core::dnn {

TrtAlgoInference::TrtAlgoInference(const AlgoConstructParams &params)
    : m_params(params.getParam<AlgoInferParams>("params")) {
  LOG_INFO_S << "TrtAlgoInference created for model: " << m_params.name;
}

TrtAlgoInference::~TrtAlgoInference() { terminate(); }

// ============================================================================
// IInferEnginePlugin Implementation
// ============================================================================

InferErrorCode TrtAlgoInference::initialize() {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_isInitialized) {
    LOG_INFO_S << "TrtAlgoInference already initialized for model: "
               << m_params.name;
    return InferErrorCode::SUCCESS;
  }

  LOG_INFO_S << "Initializing TrtAlgoInference for model: " << m_params.name;

  try {
    InferErrorCode err =
        loadEngineFromPath(m_params.model_path, m_params.need_decrypt);
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

    m_isInitialized = true;
    LOG_INFO_S << "TrtAlgoInference initialized successfully for model: "
               << m_params.name;
    LOG_INFO_S << "All inputs static: " << (m_allInputsStatic ? "yes" : "no");
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during initialization: " << e.what();
    releaseResources();
    return InferErrorCode::InitFailed;
  }
}

std::shared_ptr<IExecutionContext> TrtAlgoInference::acquireContext() {
  {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    if (!m_idlePool.empty()) {
      auto ctx = std::move(m_idlePool.back());
      m_idlePool.pop_back();
      return ctx;
    }
  }
  // Pool empty: create a fresh context outside the lock (each has its own CUDA
  // stream + buffers). Under N concurrent callers the pool naturally grows to
  // N contexts.
  return createExecutionContext();
}

void TrtAlgoInference::releaseContext(std::shared_ptr<IExecutionContext> ctx) {
  if (!ctx) {
    return;
  }
  std::lock_guard<std::mutex> lock(m_poolMutex);
  m_idlePool.push_back(std::move(ctx));
}

InferErrorCode TrtAlgoInference::infer(const TensorData &inputs,
                                       TensorData &outputs) {
  if (!m_isInitialized) {
    LOG_ERROR_S << "Inference called on uninitialized model.";
    return InferErrorCode::NotInitialized;
  }

  // Borrow an execution context (own CUDA stream + buffers) so concurrent
  // callers run in parallel instead of serializing on a global mutex. Sync
  // semantics: submit async then block on the future.
  std::shared_ptr<IExecutionContext> ctx;
  try {
    ctx = acquireContext();
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Failed to acquire execution context: " << e.what();
    return InferErrorCode::InferExecutionFailed;
  }

  InferErrorCode ret;
  try {
    ret = ctx->inferAsync(inputs, outputs).get();
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during inference: " << e.what();
    releaseContext(std::move(ctx));
    return InferErrorCode::InferFailed;
  }
  releaseContext(std::move(ctx));
  return ret;
}

const ModelInfo &TrtAlgoInference::getModelInfo() {
  if (!m_isInitialized || !m_modelInfo) {
    LOG_WARNING_S << "getModelInfo() called on uninitialized model.";
    static ModelInfo empty_info;
    return empty_info;
  }
  return *m_modelInfo;
}

InferErrorCode TrtAlgoInference::terminate() {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_isInitialized) {
    LOG_INFO_S
        << "TrtAlgoInference terminate called on uninitialized instance: "
        << m_params.name;
    return InferErrorCode::SUCCESS;
  }
  releaseResources();
  m_isInitialized = false;
  return InferErrorCode::SUCCESS;
}

// ============================================================================
// IAsyncInferEngine Implementation
// ============================================================================

std::shared_ptr<IExecutionContext> TrtAlgoInference::createExecutionContext() {
  if (!m_isInitialized) {
    throw std::runtime_error("Cannot create stream: engine not initialized");
  }

  auto stream = std::make_shared<TrtInferStream>(*this);
  auto result = stream->initialize();
  if (result != InferErrorCode::SUCCESS) {
    throw std::runtime_error("Failed to initialize inference stream");
  }

  LOG_INFO_S << "Created new inference stream for model: " << m_params.name;
  return stream;
}

TypedBuffer TrtAlgoInference::allocateAcceleratorBuffer(DataType type,
                                                        size_t size_bytes) {
  return TypedBuffer::createPinnedHost(type, size_bytes);
}

TrtAlgoInference::ContextPackage TrtAlgoInference::createContextPackage() {
  if (!m_isInitialized || !m_modelInfo) {
    throw std::runtime_error(
        "Cannot create stream context: engine not initialized");
  }

  ContextPackage ctx;
  ctx.context = createExecutionContext();

  // Pre-allocate pinned input buffers based on max sizes
  for (const auto &input : m_modelInfo->inputs) {
    size_t size_bytes = m_tensorSizeMap.at(input.name);
    std::vector<int> shape_int(input.shape.begin(), input.shape.end());
    ctx.inputs.set(input.name,
                   TypedBuffer::createPinnedHost(input.data_type, size_bytes),
                   std::move(shape_int));
  }

  // Pre-allocate pinned output buffers based on max sizes
  for (const auto &output : m_modelInfo->outputs) {
    size_t size_bytes = m_tensorSizeMap.at(output.name);
    std::vector<int> shape_int(output.shape.begin(), output.shape.end());
    ctx.outputs.set(output.name,
                    TypedBuffer::createPinnedHost(output.data_type, size_bytes),
                    std::move(shape_int));
  }

  LOG_INFO_S << "Created stream context with pre-allocated buffers";
  return ctx;
}

// ============================================================================
// Internal Implementation
// ============================================================================

void TrtAlgoInference::releaseResources() {
  LOG_INFO_S << "Releasing TensorRT resources for model: " << m_params.name;

  // Pooled execution contexts hold device resources (and keep the engine
  // alive via shared_ptr); drop them before the engine.
  {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    m_idlePool.clear();
  }

  m_context.reset();
  m_engine.reset();
  m_runtime.reset();

  if (m_stream) {
    cudaError_t err = cudaStreamDestroy(m_stream);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG_WARNING_S << "Failed to destroy CUDA stream: "
                    << cudaGetErrorString(err);
    }
    m_stream = nullptr;
  }

  m_managedBuffers.clear();
  m_pinnedOutputBuffers.clear();
  m_tensorAddressMap.clear();
  m_tensorSizeMap.clear();
  m_cachedInputShapes.clear();
  m_modelInfo.reset();

  LOG_INFO_S << "TensorRT resources released for model: " << m_params.name;
}

int64_t TrtAlgoInference::calculateVolume(const nvinfer1::Dims &dims) {
  return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1},
                         std::multiplies<int64_t>());
}

InferErrorCode TrtAlgoInference::loadEngineFromPath(const std::string &path,
                                                    bool needs_decrypt) {
  if (!std::filesystem::exists(path)) {
    LOG_ERROR_S << "Model file does not exist: " << path;
    return InferErrorCode::InitModelLoadFailed;
  }

  std::vector<char> engine_data;
  if (needs_decrypt) {
    LOG_INFO_S << "Decrypting TensorRT engine: " << path;
    std::vector<unsigned char> decrypted_data;
    auto crypto_config =
        encrypt::Crypto::deriveKeyFromCommit(m_params.decryptkey_str);
    encrypt::Crypto crypto(crypto_config);
    if (!crypto.decryptData(path, decrypted_data)) {
      LOG_ERROR_S << "Failed to decrypt model data: " << path;
      return InferErrorCode::InitDecryptionFailed;
    }
    if (decrypted_data.empty()) {
      LOG_ERROR_S << "Decryption resulted in empty model data: " << path;
      return InferErrorCode::InitModelLoadFailed;
    }
    engine_data.assign(decrypted_data.begin(), decrypted_data.end());
  } else {
    std::ifstream engine_file(path, std::ios::binary);
    if (!engine_file) {
      LOG_ERROR_S << "Failed to open TensorRT engine file: " << path;
      return InferErrorCode::InitModelLoadFailed;
    }
    engine_file.seekg(0, std::ios::end);
    size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    engine_data.resize(file_size);
    engine_file.read(engine_data.data(), file_size);
  }

  if (engine_data.empty()) {
    LOG_ERROR_S << "Engine data is empty for model: " << path;
    return InferErrorCode::InitModelLoadFailed;
  }

  m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
  if (!m_runtime) {
    LOG_ERROR_S << "Failed to create TensorRT Runtime.";
    return InferErrorCode::InitRuntimeFailed;
  }

  m_engine.reset(
      m_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!m_engine) {
    LOG_ERROR_S << "Failed to deserialize TensorRT engine.";
    return InferErrorCode::InitEngineFailed;
  }

  LOG_INFO_S << "TensorRT engine loaded and deserialized successfully: "
             << m_params.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::setupBindings() {
  m_context.reset(m_engine->createExecutionContext());
  if (!m_context) {
    LOG_ERROR_S << "Failed to create TensorRT Execution Context.";
    return InferErrorCode::InitContextFailed;
  }

  int least_priority, greatest_priority;
  CHECK_CUDA_ERROR(
      cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(
      &m_stream, cudaStreamNonBlocking, greatest_priority));

  m_managedBuffers.clear();
  m_tensorAddressMap.clear();
  m_tensorSizeMap.clear();
  m_dynamicInputTensorNames.clear();
  m_cachedInputShapes.clear();
  m_allInputsStatic = true;

  m_modelInfo = std::make_shared<ModelInfo>();
  m_modelInfo->name = m_params.name;

  const int profile_index = 0;
  if (m_engine->getNbOptimizationProfiles() <= profile_index) {
    LOG_ERROR_S << "Engine does not have optimization profile at index "
                << profile_index;
    return InferErrorCode::InitFailed;
  }

  const int32_t num_io_tensors = m_engine->getNbIOTensors();

  // Set input shapes to MAX for buffer allocation
  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char *name = m_engine->getIOTensorName(i);
    if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      auto max_dims = m_engine->getProfileShape(
          name, profile_index, nvinfer1::OptProfileSelector::kMAX);
      if (!m_context->setInputShape(name, max_dims)) {
        LOG_WARNING_S
            << "Failed to set max input shape for auto-sizing tensor: " << name;
      }
    }
  }

  m_managedBuffers.reserve(num_io_tensors);

  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char *name = m_engine->getIOTensorName(i);
    auto trt_dtype = m_engine->getTensorDataType(name);

    auto dims = m_engine->getProfileShape(name, profile_index,
                                          nvinfer1::OptProfileSelector::kMAX);

    int64_t volume = -1;
    size_t buffer_size = 0;

    if (dims.nbDims >= 0) {
      volume = calculateVolume(dims);
    }

    if (volume < 0) {
      if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        auto it = m_params.max_output_buffer_sizes.find(name);
        if (it != m_params.max_output_buffer_sizes.end()) {
          buffer_size = it->second;
        } else {
          nvinfer1::Dims deduced_dims = m_context->getTensorShape(name);
          int64_t deduced_volume = calculateVolume(deduced_dims);

          if (deduced_volume > 0) {
            buffer_size = static_cast<size_t>(deduced_volume) *
                          trt_utils::getTrtElementSize(trt_dtype);
          } else {
            LOG_ERROR_S << "Could not deduce max size for dynamic output: "
                        << name;
            return InferErrorCode::InitBindingFailed;
          }
        }
      } else {
        LOG_ERROR_S << "Input tensor '" << name
                    << "' has unexpected dynamic dimension.";
        return InferErrorCode::InitBindingFailed;
      }
    } else {
      buffer_size =
          static_cast<size_t>(volume) * trt_utils::getTrtElementSize(trt_dtype);
    }

    m_managedBuffers.emplace_back(cuda_utils::DeviceByteBuffer{buffer_size});
    void *device_ptr = m_managedBuffers.back().unsafePtr();

    m_tensorAddressMap[name] = device_ptr;
    m_tensorSizeMap[name] = buffer_size;

    if (!m_context->setTensorAddress(name, device_ptr)) {
      LOG_ERROR_S << "Failed to set tensor address for: " << name;
      return InferErrorCode::InitBindingFailed;
    }

    // Populate ModelInfo
    ModelInfo::TensorInfo tensor_info;
    tensor_info.name = name;

    auto binding_dims = m_engine->getTensorShape(name);
    tensor_info.shape.assign(binding_dims.d,
                             binding_dims.d + binding_dims.nbDims);
    tensor_info.data_type = trt_utils::trtDataTypeToAiCore(trt_dtype);

    if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      for (const auto &dim : tensor_info.shape) {
        if (dim == -1) {
          m_dynamicInputTensorNames.insert(name);
          m_allInputsStatic = false;
          LOG_DEBUG_S << "Input tensor '" << name
                      << "' is identified as dynamic.";
          break;
        }
      }
      m_modelInfo->inputs.emplace_back(std::move(tensor_info));
    } else {
      m_modelInfo->outputs.emplace_back(std::move(tensor_info));
    }
  }

  LOG_INFO_S << "Bindings and buffers configured for model: " << m_params.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::setupPinnedOutputBuffers() {
  for (const auto &output_info : m_modelInfo->outputs) {
    const auto &name = output_info.name;
    size_t buffer_size = m_tensorSizeMap.at(name);

    cuda_utils::CudaHostBuffer<uint8_t> pinned_buffer;
    pinned_buffer.reserve(buffer_size);

    m_pinnedOutputBuffers[name] = std::move(pinned_buffer);

    LOG_DEBUG_S << "Pre-allocated " << buffer_size
                << " bytes of pinned memory for output: " << name;
  }

  LOG_INFO_S << "Pinned output buffers allocated for "
             << m_modelInfo->outputs.size() << " outputs.";
  return InferErrorCode::SUCCESS;
}

bool TrtAlgoInference::updateInputShapesIfNeeded(const TensorData &inputs) {
  for (const auto &input_info : m_modelInfo->inputs) {
    const auto &name = input_info.name;
    const bool is_dynamic = m_dynamicInputTensorNames.count(name);

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
      LOG_TRACE_S << "Updated input shape for tensor '" << name << "'";
    }
  }

  return true;
}

void TrtAlgoInference::copyInputsToDevice(const TensorData &inputs) {
  for (const auto &input_info : m_modelInfo->inputs) {
    const auto &name = input_info.name;
    const TypedBuffer &input_buffer = inputs.at(name).buffer;
    const size_t actual_size_bytes = input_buffer.getSizeBytes();
    void *dest_device_ptr = m_tensorAddressMap.at(name);

    if (input_buffer.location() == BufferLocation::CPU) {
      const void *src_host_ptr = input_buffer.getRawHostPtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dest_device_ptr, src_host_ptr,
                                       actual_size_bytes,
                                       cudaMemcpyHostToDevice, m_stream));
    } else if (input_buffer.location() == BufferLocation::GpuDevice) {
      void *src_device_ptr = input_buffer.getRawDevicePtr();
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dest_device_ptr, src_device_ptr,
                                       actual_size_bytes,
                                       cudaMemcpyDeviceToDevice, m_stream));
    }
  }
}

void TrtAlgoInference::copyOutputsToHost(TensorData &outputs) {
  outputs.clear();

  for (const auto &output_info : m_modelInfo->outputs) {
    const auto &name = output_info.name;
    void *src_device_ptr = m_tensorAddressMap.at(name);

    nvinfer1::Dims actual_output_dims = m_context->getTensorShape(name.c_str());
    int64_t actual_volume = calculateVolume(actual_output_dims);

    size_t actual_output_size_bytes =
        static_cast<size_t>(actual_volume) *
        trt_utils::getTrtElementSize(
            trt_utils::aiCoreDataTypeToTrt(output_info.data_type));

    auto &pinned_buffer = m_pinnedOutputBuffers.at(name);

    if (pinned_buffer.capacity() < actual_output_size_bytes) {
      pinned_buffer.reserve(actual_output_size_bytes);
    }

    uint8_t *dest_host_ptr = pinned_buffer.writePtr(actual_output_size_bytes);

    CHECK_CUDA_ERROR(cudaMemcpyAsync(dest_host_ptr, src_device_ptr,
                                     actual_output_size_bytes,
                                     cudaMemcpyDeviceToHost, m_stream));

    outputs.set(
        name, TypedBuffer(),
        std::vector<int>(actual_output_dims.d,
                         actual_output_dims.d + actual_output_dims.nbDims));
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

  for (const auto &output_info : m_modelInfo->outputs) {
    const auto &name = output_info.name;
    auto &pinned_buffer = m_pinnedOutputBuffers.at(name);
    std::vector<uint8_t> safe_data = pinned_buffer.toVector();
    outputs.find(name)->buffer =
        TypedBuffer::createFromCpu(output_info.data_type, std::move(safe_data));
  }
}

InferErrorCode TrtAlgoInference::inferWithoutGraph(const TensorData &inputs,
                                                   TensorData &outputs) {
  copyInputsToDevice(inputs);

  LOG_TRACE_S << "Executing inference on stream " << m_stream;
  if (!m_context->enqueueV3(m_stream)) {
    LOG_ERROR_S << "Failed to enqueue TensorRT inference.";
    return InferErrorCode::InferExecutionFailed;
  }

  copyOutputsToHost(outputs);

  return InferErrorCode::SUCCESS;
}

} // namespace ai_core::dnn