/**
 * @file trt_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_TENSORRT_INFERENCE_HPP
#define AI_CORE_TENSORRT_INFERENCE_HPP

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_async.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/logger.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_host_buffer.cuh"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_core::dnn {

// Forward declaration
class TrtInferStream;

class TrtFrameworkLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      LOG_FATALS << "[TRT] " << msg;
      break;
    case Severity::kERROR:
      LOG_ERROR_S << "[TRT] " << msg;
      break;
    case Severity::kWARNING:
      LOG_WARNING_S << "[TRT] " << msg;
      break;
    case Severity::kINFO:
      LOG_INFO_S << "[TRT] " << msg;
      break;
    case Severity::kVERBOSE:
      LOG_DEBUG_S << "[TRT] " << msg;
      break;
    default:
      LOG_INFO_S << "[TRT] " << msg;
      break;
    }
  }
};

/**
 * @brief TensorRT-based inference engine with async capabilities
 *
 * This class now extends IAsyncInferEngine to provide both synchronous
 * (via base class infer()) and asynchronous (via IExecutionContext) inference.
 *
 * Key Design Changes:
 * - Inherits IAsyncInferEngine instead of IInferEnginePlugin
 * - infer() uses inferWithoutGraph for backward compatibility (single-threaded
 * sync)
 * - createExecutionContext() creates independent TrtInferStream with isolated
 * resources
 * - CUDA Graph is only available through TrtInferStream::setGraphEnabled()
 *
 * Resource Ownership:
 * - Shared (owned by TrtAlgoInference):
 *   - IRuntime, ICudaEngine, ModelInfo
 * - Per-Stream (owned by TrtInferStream):
 *   - IExecutionContext, cudaStream_t, Device Buffers, CUDA Graph
 *
 * Thread Safety:
 * - infer(): Thread-safe via internal mutex (uses dedicated default stream)
 * - createExecutionContext(): Thread-safe, returns independent stream
 * - Each TrtInferStream: NOT thread-safe, use one per thread
 */
class TrtAlgoInference : public IAsyncInferEngine {
public:
  explicit TrtAlgoInference(const AlgoConstructParams &params);
  ~TrtAlgoInference() override;

  // Non-copyable, non-movable
  TrtAlgoInference(const TrtAlgoInference &) = delete;
  TrtAlgoInference &operator=(const TrtAlgoInference &) = delete;
  TrtAlgoInference(TrtAlgoInference &&) = delete;
  TrtAlgoInference &operator=(TrtAlgoInference &&) = delete;

  // ============================================================================
  // IInferEnginePlugin Interface (Backward Compatible Sync Mode)
  // ============================================================================

  InferErrorCode initialize() override;

  /**
   * @brief Synchronous inference (backward compatible)
   *
   * Uses a dedicated default stream internally, always uses inferWithoutGraph.
   * Thread-safe via internal mutex.
   *
   * @note For multi-threaded or high-performance scenarios, use
   *       createExecutionContext() + IExecutionContext::inferAsync() instead.
   */
  InferErrorCode infer(const TensorData &inputs, TensorData &outputs) override;

  const ModelInfo &getModelInfo() override;
  InferErrorCode terminate() override;

  // ============================================================================
  // IAsyncInferEngine Interface (New Async Capabilities)
  // ============================================================================

  /**
   * @brief Create an independent inference stream
   *
   * Each stream has its own:
   * - cudaStream_t
   * - IExecutionContext
   * - Device buffers (managed by CudaDeviceBuffer)
   * - Pinned output buffers (managed by CudaHostBuffer)
   * - CUDA Graph resources (optional, via setGraphEnabled)
   *
   * @return std::shared_ptr<IExecutionContext> A new independent stream
   * @throws std::runtime_error if engine not initialized
   */
  std::shared_ptr<IExecutionContext> createExecutionContext() override;

  /**
   * @brief Allocate pinned host memory
   *
   * Uses cudaMallocHost internally. The returned TypedBuffer has:
   * - location = BufferLocation::CPU
   * - memoryType = BufferMemoryType::Pinned
   *
   * @param type Data type
   * @param sizeBytes Size in bytes
   * @return TypedBuffer backed by pinned memory
   */
  TypedBuffer allocateAcceleratorBuffer(DataType type,
                                        size_t sizeBytes) override;

  /**
   * @brief Create stream with pre-allocated pinned I/O buffers
   *
   * Allocates pinned buffers for all model inputs/outputs based on
   * max shapes from optimization profile.
   */
  ContextPackage createContextPackage() override;

private:
  friend class TrtInferStream;

  // ============================================================================
  // Initialization Helpers
  // ============================================================================

  static int64_t calculateVolume(const nvinfer1::Dims &dims);
  InferErrorCode loadEngineFromPath(const std::string &path,
                                    bool needs_decrypt);
  InferErrorCode setupBindings();
  InferErrorCode setupPinnedOutputBuffers();
  void releaseResources();

  // ============================================================================
  // Sync Mode Inference (for backward compatible infer())
  // ============================================================================

  /**
   * @brief Standard sync inference without CUDA Graph
   *
   * Used by infer() for backward compatibility.
   */
  InferErrorCode inferWithoutGraph(const TensorData &inputs,
                                   TensorData &outputs);

  bool updateInputShapesIfNeeded(const TensorData &inputs);
  void copyInputsToDevice(const TensorData &inputs);
  void copyOutputsToHost(TensorData &outputs);

  // ============================================================================
  // Member Variables - Shared Resources
  // ============================================================================

  AlgoInferParams mParams;
  TrtFrameworkLogger mLogger;

  // TRT core components (shared across all streams)
  std::unique_ptr<nvinfer1::IRuntime> mRuntime;
  // Use shared_ptr so that streams can keep the engine alive
  // This prevents "destroying engine before context" errors
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

  // Model metadata (shared, read-only after init)
  std::shared_ptr<ModelInfo> modelInfo;

  // Binding metadata (shared, read-only after init)
  std::unordered_set<std::string> mDynamicInputTensorNames;
  std::unordered_map<std::string, size_t> mTensorSizeMap;
  bool mAllInputsStatic{true};

  bool mIsInitialized{false};

  // ============================================================================
  // Member Variables - Default Stream Resources (for sync infer())
  // ============================================================================

  // Execution context for default sync stream
  std::unique_ptr<nvinfer1::IExecutionContext> mContext;

  // CUDA stream for default sync path
  cudaStream_t mStream{nullptr};

  // Device buffers for default sync path
  std::vector<cuda_utils::DeviceByteBuffer> mManagedBuffers;
  std::unordered_map<std::string, void *> mTensorAddressMap;

  // Pinned output buffers for default sync path
  std::unordered_map<std::string, cuda_utils::CudaHostBuffer<uint8_t>>
      mPinnedOutputBuffers;

  // Cached input shapes (to avoid redundant setInputShape calls)
  std::unordered_map<std::string, std::vector<int64_t>> mCachedInputShapes;

  // Mutex for thread-safe sync infer()
  mutable std::mutex mMutex;
};

} // namespace ai_core::dnn

#endif // AI_CORE_TENSORRT_INFERENCE_HPP