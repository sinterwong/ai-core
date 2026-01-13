/**
 * @file gpu_generic_cuda_preprocessor.cuh
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief GPU-accelerated frame preprocessor with optimized memory management
 * @version 0.2
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 * Design improvements:
 * 1. Pre-allocated buffers for model parameters (mean/std/pad) - initialized
 * once
 * 2. Cached working buffers (HWC/CHW/output) - sized to max expected usage
 * 3. Dedicated CUDA stream for async execution
 * 4. Lazy initialization on first use to avoid breaking the interface contract
 */
#ifndef GPU_GENERIC_CUDA_PREPROCESSOR_HPP
#define GPU_GENERIC_CUDA_PREPROCESSOR_HPP

#include "ai_core/input_types.hpp"
#include "ai_core/preprocess_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_stream.cuh"
#include "cuda_utils.hpp"
#include "preproc/frame_preprocessor_base.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace ai_core::dnn::gpu {
/**
 * @brief Configuration for GpuGenericCudaPreprocessor
 */
struct GpuPreprocessorConfig {
  /// Enable parallel mode (each call allocates own memory, thread-safe)
  /// When false, uses cached buffers with mutex protection
  bool enable_parallel = false;

  /// Use high-priority CUDA stream (only effective in sequential mode)
  bool use_high_priority_stream = false;

  static GpuPreprocessorConfig defaults() { return GpuPreprocessorConfig{}; }

  /// Create config for parallel/multi-threaded usage
  static GpuPreprocessorConfig parallel() {
    GpuPreprocessorConfig cfg;
    cfg.enable_parallel = true;
    return cfg;
  }
};

/**
 * @brief GPU-accelerated frame preprocessor
 *
 * Thread Safety:
 * - Sequential mode (enableParallel=false): Thread-safe via mutex, serialized
 * - Parallel mode (enableParallel=true): Thread-safe, each call independent
 */
class GpuGenericCudaPreprocessor : public IFramePreprocessor {
public:
  using Config = GpuPreprocessorConfig;

  GpuGenericCudaPreprocessor();
  explicit GpuGenericCudaPreprocessor(const Config &config);
  ~GpuGenericCudaPreprocessor() override;

  GpuGenericCudaPreprocessor(const GpuGenericCudaPreprocessor &) = delete;
  GpuGenericCudaPreprocessor &
  operator=(const GpuGenericCudaPreprocessor &) = delete;
  GpuGenericCudaPreprocessor(GpuGenericCudaPreprocessor &&) = delete;
  GpuGenericCudaPreprocessor &operator=(GpuGenericCudaPreprocessor &&) = delete;

  TypedBuffer process(const FramePreprocessArg &args, const FrameInput &input,
                      FrameTransformContext &runtime_args) const override;

  TypedBuffer
  batchProcess(const FramePreprocessArg &args,
               const std::vector<FrameInput> &inputs,
               std::vector<FrameTransformContext> &runtime_args) const override;

  /// Get the CUDA stream (only valid in sequential mode)
  cudaStream_t getStream() const;

  /// Synchronize the stream (only valid in sequential mode)
  void synchronize() const;

  /// Reset cached resources (only effective in sequential mode)
  void resetCache() const;

  /// Check if running in parallel mode
  bool isParallelMode() const { return m_config.enable_parallel; }

private:
  TypedBuffer processSequential(const FramePreprocessArg &args,
                                const FrameInput &input,
                                FrameTransformContext &runtime_args) const;

  TypedBuffer batchProcessSequential(
      const FramePreprocessArg &args, const std::vector<FrameInput> &inputs,
      std::vector<FrameTransformContext> &runtime_args) const;

  TypedBuffer processParallel(const FramePreprocessArg &args,
                              const FrameInput &input,
                              FrameTransformContext &runtime_args) const;

  TypedBuffer
  batchProcessParallel(const FramePreprocessArg &args,
                       const std::vector<FrameInput> &inputs,
                       std::vector<FrameTransformContext> &runtime_args) const;

  static void validatePreprocessArgs(const FramePreprocessArg &args,
                                     int src_channels);

  struct CachedResources {
    // Parameter buffers
    cuda_utils::CudaDeviceBuffer<float> d_mean;
    cuda_utils::CudaDeviceBuffer<float> d_std;
    cuda_utils::CudaDeviceBuffer<int> d_pad;

    // Host-side copies for change detection
    std::vector<float> cached_mean_vals;
    std::vector<float> cached_norm_vals;
    std::vector<int> cached_pad_vals;

    // Working buffers
    cuda_utils::DeviceByteBuffer d_hwc_buffer;
    cuda_utils::DeviceByteBuffer d_chw_buffer;

    // Input image buffer (reused across calls to avoid alloc/free)
    cuda_utils::DeviceByteBuffer d_input_image;

    // Batch processing: input image buffers (one per batch slot)
    std::vector<cuda_utils::DeviceByteBuffer> d_batch_input_images;

    // Batch metadata buffers
    cuda_utils::CudaDeviceBuffer<uint8_t *> d_src_ptrs;
    cuda_utils::CudaDeviceBuffer<int> d_src_heights;
    cuda_utils::CudaDeviceBuffer<int> d_src_widths;
    cuda_utils::CudaDeviceBuffer<cuda_op::ROIData> d_rois;
    cuda_utils::CudaDeviceBuffer<int> d_new_heights;
    cuda_utils::CudaDeviceBuffer<int> d_new_widths;
    cuda_utils::CudaDeviceBuffer<int> d_pad_ys;
    cuda_utils::CudaDeviceBuffer<int> d_pad_xs;

    void reset();
  };

  void updateParameterBuffers(const FramePreprocessArg &args,
                              cudaStream_t stream) const;
  void ensureWorkingBufferCapacity(const FramePreprocessArg &args,
                                   int batch_size, cudaStream_t stream) const;

  // Stream for sequential mode
  mutable std::unique_ptr<cuda_utils::CudaStream> m_stream;

  // Cached resources for sequential mode
  mutable CachedResources m_cache;
  mutable std::mutex m_mutex;

  Config m_config;
};

} // namespace ai_core::dnn::gpu

#endif