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
/** Configures allocation and scheduling for `GpuGenericCudaPreprocessor`. */
struct GpuPreprocessorConfig {
  /** Give each call independent buffers instead of using the shared cache. */
  bool enable_parallel = false;

  /** Use a high-priority stream in cached-buffer mode. */
  bool use_high_priority_stream = false;

  static GpuPreprocessorConfig defaults() { return GpuPreprocessorConfig{}; }

  static GpuPreprocessorConfig parallel() {
    GpuPreprocessorConfig cfg;
    cfg.enable_parallel = true;
    return cfg;
  }
};

/**
 * Preprocesses frames on a CUDA device.
 *
 * @par Thread safety
 * Cached-buffer mode serializes calls with a mutex. Parallel mode gives each
 * call independent buffers and permits concurrent execution.
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

  /** Returns the owned stream, or `nullptr` in parallel mode. */
  cudaStream_t getStream() const;

  /** Waits for cached-buffer work; has no effect in parallel mode. */
  void synchronize() const;

  /**
   * Releases cached buffers after synchronizing; has no effect in parallel
   * mode.
   */
  void resetCache() const;

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
    cuda_utils::CudaDeviceBuffer<float> d_mean;
    cuda_utils::CudaDeviceBuffer<float> d_std;
    cuda_utils::CudaDeviceBuffer<int> d_pad;

    // Host copies avoid uploading unchanged normalization parameters.
    std::vector<float> cached_mean_vals;
    std::vector<float> cached_std_vals;
    std::vector<int> cached_pad_vals;

    cuda_utils::DeviceByteBuffer d_hwc_buffer;
    cuda_utils::DeviceByteBuffer d_chw_buffer;

    cuda_utils::DeviceByteBuffer d_input_image;

    std::vector<cuda_utils::DeviceByteBuffer> d_batch_input_images;

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

  mutable std::unique_ptr<cuda_utils::CudaStream> m_stream;

  mutable CachedResources m_cache;
  mutable std::mutex m_mutex;

  Config m_config;
};

} // namespace ai_core::dnn::gpu

#endif
