/**
 * @file cuda_utils.cu
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "cuda_utils.hpp"
#include <cuda_fp16.h>

namespace ai_core::dnn::gpu::cuda_op {

__global__ void hwc_to_chw_kernel(const float *src, float *dst, int height,
                                  int width, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    for (int c = 0; c < channels; ++c) {
      dst[c * (width * height) + y * width + x] =
          src[y * width * channels + x * channels + c];
    }
  }
}

void hwc_to_chw_gpu(const float *src, float *dst, int height, int width,
                    int channels) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  hwc_to_chw_kernel<<<grid, block>>>(src, dst, height, width, channels);
}

__global__ void fp32_to_fp16_kernel(const float *src, uint16_t *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = __float2half_rn(src[i]);
  }
}

void fp32_to_fp16_gpu(const float *src, uint16_t *dst, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fp32_to_fp16_kernel<<<blocks, threads>>>(src, dst, n);
}

__global__ void crop_resize_normalize_kernel(const uint8_t *src, float *dst,
                                             int src_h, int src_w, int src_c,
                                             int crop_x, int crop_y, int crop_h,
                                             int crop_w, int dst_h, int dst_w,
                                             const float *mean,
                                             const float *std) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dst_w && y < dst_h) {
    float src_x = (float)(x) * (float)(crop_w) / (float)(dst_w) + crop_x;
    float src_y = (float)(y) * (float)(crop_h) / (float)(dst_h) + crop_y;

    int x1 = floorf(src_x);
    int y1 = floorf(src_y);

    for (int c = 0; c < src_c; c++) {
      float p1 = src[(y1 * src_w + x1) * src_c + c];
      dst[(y * dst_w + x) * src_c + c] = (p1 - mean[c]) / std[c];
    }
  }
}

void crop_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                               int src_w, int src_c, int crop_x, int crop_y,
                               int crop_h, int crop_w, int dst_h, int dst_w,
                               const float *mean, const float *std) {
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  crop_resize_normalize_kernel<<<grid, block>>>(src, dst, src_h, src_w, src_c,
                                                crop_x, crop_y, crop_h, crop_w,
                                                dst_h, dst_w, mean, std);
}

__global__ void escale_resize_normalize_kernel(
    const uint8_t *src, float *dst, int src_h, int src_w, int src_c, int dst_h,
    int dst_w, const float *mean, const float *std, const float *pad_val) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dst_w && y < dst_h) {
    float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;

    if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h) {
      float src_x = (float)(x - pad_x) / scale;
      float src_y = (float)(y - pad_y) / scale;

      int x1 = floorf(src_x);
      int y1 = floorf(src_y);

      for (int c = 0; c < src_c; c++) {
        float p1 = src[(y1 * src_w + x1) * src_c + c];
        dst[(y * dst_w + x) * src_c + c] = (p1 - mean[c]) / std[c];
      }
    } else {
      for (int c = 0; c < src_c; c++) {
        dst[(y * dst_w + x) * src_c + c] = pad_val[c];
      }
    }
  }
}

void escale_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                                 int src_w, int src_c, int dst_h, int dst_w,
                                 const float *mean, const float *std,
                                 const float *pad_val) {
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  escale_resize_normalize_kernel<<<grid, block>>>(
      src, dst, src_h, src_w, src_c, dst_h, dst_w, mean, std, pad_val);
}
} // namespace ai_core::dnn::gpu::cuda_op
