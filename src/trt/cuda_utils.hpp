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

namespace ai_core::dnn::gpu::cuda_op {

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
} // namespace ai_core::dnn::gpu::cuda_op
#endif
