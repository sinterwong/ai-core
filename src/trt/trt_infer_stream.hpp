/**
 * @file trt_infer_stream.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief TensorRT inference stream with independent resources and CUDA Graph
 * @version 0.1
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_TRT_INFER_STREAM_HPP
#define AI_CORE_TRT_INFER_STREAM_HPP

#include "ai_core/infer_stream.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_host_buffer.cuh"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <future>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ai_core::dnn {

// Forward declaration
class TrtAlgoInference;

/**
 * @brief TensorRT inference stream implementation
 *
 * Each TrtInferStream instance owns independent resources:
 * - CUDA stream (cudaStream_t)
 * - TensorRT execution context (IExecutionContext)
 * - Device memory buffers (CudaDeviceBuffer for each I/O tensor)
 * - Pinned output buffers (CudaHostBuffer for each output)
 * - CUDA Graph resources (optional, enabled via setGraphEnabled)
 *
 * This design solves the multi-threading CUDA Graph crash issue by ensuring:
 * 1. Each thread has its own stream with isolated resources
 * 2. CUDA Graph instances are bound to specific buffer addresses (per-stream)
 * 3. No resource contention between threads
 *
 * Thread Safety:
 * - Each TrtInferStream should be used by only ONE thread
 * - Multiple TrtInferStreams can run concurrently on different threads
 * - The parent TrtAlgoInference is shared but only provides read-only data
 *
 * CUDA Graph Lifecycle:
 * 1. setGraphEnabled(true) - enables graph mode
 * 2. First inferAsync() - captures the graph
 * 3. Subsequent inferAsync() - replays the captured graph
 * 4. setGraphEnabled(false) - destroys graph resources
 *
 * @warning Input shapes must remain constant after graph capture.
 *          Changing shapes will invalidate the graph.
 */
class TrtInferStream : public IInferStream {
public:
  /**
   * @brief Construct a new TRT inference stream
   *
   * @param engine Parent engine (provides ICudaEngine, ModelInfo)
   */
  explicit TrtInferStream(TrtAlgoInference &engine);

  ~TrtInferStream() override;

  // Non-copyable, non-movable
  TrtInferStream(const TrtInferStream &) = delete;
  TrtInferStream &operator=(const TrtInferStream &) = delete;
  TrtInferStream(TrtInferStream &&) = delete;
  TrtInferStream &operator=(TrtInferStream &&) = delete;

  // ============================================================================
  // IInferStream Interface
  // ============================================================================

  /**
   * @brief Asynchronously submit inference
   *
   * Data flow:
   * 1. Copy inputs from host to device (async if pinned)
   * 2. Update input shapes if dynamic
   * 3. Execute inference (enqueueV3 or graph launch)
   * 4. Copy outputs from device to host (async)
   * 5. Return future that resolves after completion
   *
   * @param inputs Input tensors (Pinned memory recommended)
   * @param outputs Output tensors (populated after future.get())
   * @return Future with inference result
   */
  std::future<InferErrorCode> inferAsync(const TensorData &inputs,
                                         TensorData &outputs) override;

  InferErrorCode synchronize() override;

  bool isComplete() const override;

  StreamHandle getHandle() const override;

  /**
   * @brief Enable/disable CUDA Graph
   *
   * When enabling:
   * - Next inferAsync() will capture operations into a graph
   * - Subsequent calls replay the graph for better performance
   *
   * When disabling:
   * - Destroys captured graph resources
   * - Returns to normal enqueueV3 execution
   *
   * @note Graph is only available when ALL input tensors have static shapes.
   *       For dynamic shapes, this will log a warning and remain disabled.
   */
  InferErrorCode setGraphEnabled(bool enable) override;

  bool isGraphEnabled() const override;

  // ============================================================================
  // Stream Lifecycle
  // ============================================================================

  /**
   * @brief Initialize stream resources
   *
   * Creates:
   * - CUDA stream (high priority, non-blocking)
   * - TensorRT execution context
   * - Device buffers for all I/O tensors
   * - Pinned output buffers
   *
   * Called automatically by TrtAlgoInference::createStream().
   *
   * @return InferErrorCode
   */
  InferErrorCode initialize();

private:
  // ============================================================================
  // Internal Methods
  // ============================================================================

  InferErrorCode allocateBuffers();
  InferErrorCode allocatePinnedOutputBuffers();

  bool updateInputShapesIfNeeded(const TensorData &inputs);
  InferErrorCode copyInputsToDevice(const TensorData &inputs);
  InferErrorCode copyOutputsToHost(TensorData &outputs);

  InferErrorCode executeInference();
  InferErrorCode captureGraph();
  InferErrorCode launchGraph();
  void destroyGraph();

  // ============================================================================
  // Member Variables
  // ============================================================================

  // Parent engine reference (provides ModelInfo, mTensorSizeMap etc.)
  TrtAlgoInference &mEngine;

  // Keep engine alive - prevents destruction while this stream exists
  // IMPORTANT: Declared BEFORE mContext so mContext is destroyed FIRST
  // (C++ destroys members in reverse declaration order)
  std::shared_ptr<nvinfer1::ICudaEngine> mSharedEngine;

  // CUDA stream (owned by this stream instance)
  cudaStream_t mCudaStream{nullptr};

  // TensorRT execution context (owned by this stream instance)
  // Will be destroyed BEFORE mSharedEngine (since it's declared after)
  std::unique_ptr<nvinfer1::IExecutionContext> mContext;

  // Device buffers for I/O tensors (owned by this stream instance)
  std::vector<cuda_utils::DeviceByteBuffer> mDeviceBuffers;
  std::unordered_map<std::string, void *> mTensorAddressMap;

  // Pinned output buffers (owned by this stream instance)
  std::unordered_map<std::string, cuda_utils::CudaHostBuffer<uint8_t>>
      mPinnedOutputBuffers;

  // Cached input shapes for this stream
  std::unordered_map<std::string, std::vector<int64_t>> mCachedInputShapes;

  // CUDA Graph resources (owned by this stream instance)
  bool mGraphEnabled{false};
  bool mGraphCaptured{false};
  cudaGraph_t mCudaGraph{nullptr};
  cudaGraphExec_t mCudaGraphExec{nullptr};

  bool mInitialized{false};
};

} // namespace ai_core::dnn

#endif // AI_CORE_TRT_INFER_STREAM_HPP
