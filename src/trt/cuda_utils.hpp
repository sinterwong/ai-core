/**
 * @file cuda_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief CUDA preprocessing kernels with stream support
 * @version 0.2
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_CUDA_UTILS_CUH
#define AI_CORE_CUDA_UTILS_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ai_core::dnn::gpu::cuda_op {

struct ROIData {
  int x, y, h, w;
};

// ===========================================================================
// Single frame operations (with stream support)
// ===========================================================================

void hwcToChwGpu(const float *src, float *dst, int height, int width,
                 int channels, cudaStream_t stream = nullptr);

void fp32ToFp16Gpu(const float *src, uint16_t *dst, int n,
                   cudaStream_t stream = nullptr);

void cropResizeNormalizeGpu(const uint8_t *src, float *dst, int src_h,
                            int src_w, int src_c, int crop_x, int crop_y,
                            int crop_h, int crop_w, int dst_h, int dst_w,
                            const float *mean, const float *std,
                            cudaStream_t stream = nullptr);

void escaleResizeNormalizeGpu(const uint8_t *src, float *dst, int full_image_w,
                              int src_c, const ROIData &roi, int dst_h,
                              int dst_w, const float *mean, const float *std,
                              const int *pad_val,
                              cudaStream_t stream = nullptr);

// ===========================================================================
// Batch operations (with stream support)
// ===========================================================================

void batchHwcToChwGpu(const float *src, float *dst, int height, int width,
                      int channels, int batch_size,
                      cudaStream_t stream = nullptr);

void batchCropResizeNormalizeGpu(const uint8_t *const *d_src_ptrs,
                                 float *d_batch_dst, const int *d_src_hs,
                                 const int *d_src_ws, int src_c,
                                 const ROIData *d_rois, int dst_h, int dst_w,
                                 const float *mean, const float *std,
                                 int batch_size, cudaStream_t stream = nullptr);

void batchEscaleResizeNormalizeGpu(
    const uint8_t *const *d_src_ptrs, float *d_batch_dst, const int *d_src_hs,
    const int *d_src_ws, int src_c, const ROIData *d_rois, int dst_h, int dst_w,
    const float *mean, const float *std, const int *pad_val,
    const int *d_new_hs, const int *d_new_ws, const int *d_pad_ys,
    const int *d_pad_xs, int batch_size, cudaStream_t stream = nullptr);

} // namespace ai_core::dnn::gpu::cuda_op

#endif
