/**
 * @file cuda_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __CUDA_KERNELS_UTILS_CUH__
#define __CUDA_KERNELS_UTILS_CUH__

#include <cstdint>

namespace ai_core::dnn::gpu::cuda_op
{
    struct ROIData
    {
        int x, y, h, w;
    };

    void hwc_to_chw_gpu(const float *src, float *dst, int height, int width,
                        int channels);

    void fp32_to_fp16_gpu(const float *src, uint16_t *dst, int n);

    void crop_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                                   int src_w, int src_c, int crop_x, int crop_y,
                                   int crop_h, int crop_w, int dst_h, int dst_w,
                                   const float *mean, const float *std);

    void escale_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                                     int src_w, int src_c, int dst_h, int dst_w,
                                     const float *mean, const float *std,
                                     const float *pad_val);

    void batch_hwc_to_chw_gpu(const float *src, float *dst, int height, int width,
                              int channels, int batch_size);

    void batch_crop_resize_normalize_gpu(
        const uint8_t *const *d_src_ptrs, float *d_batch_dst,
        const int *d_src_hs, const int *d_src_ws, int src_c,
        const ROIData *d_rois,
        int dst_h, int dst_w, const float *mean, const float *std, int batch_size);

    void batch_escale_resize_normalize_gpu(
        const uint8_t *const *d_src_ptrs, float *d_batch_dst,
        const int *d_src_hs, const int *d_src_ws, int src_c,
        const ROIData *d_rois,
        int dst_h, int dst_w, const float *mean, const float *std, const float *pad_val,
        const int *d_new_hs, const int *d_new_ws,
        const int *d_pad_ys, const int *d_pad_xs,
        int batch_size);
} // namespace ai_core::dnn::gpu::cuda_op
#endif
