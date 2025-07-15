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

namespace kernel {
#define FP16_MAX_VAL 65504.0f

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

__global__ void fp32_to_fp16_clamp_kernel(const float *src, uint16_t *dst,
                                          int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // clamp the numerical values to the representational range of fp16
    float val = fminf(fmaxf(src[i], -FP16_MAX_VAL), FP16_MAX_VAL);
    dst[i] = __float2half_rn(val);
  }
}

// crop + resize + normal
__global__ void crop_resize_normalize_bilinear_kernel(
    const uint8_t *src, float *dst, int src_h, int src_w, int src_c, int crop_x,
    int crop_y, int crop_h, int crop_w, int dst_h, int dst_w, const float *mean,
    const float *std) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dst_w && y < dst_h) {
    float src_x = (x + 0.5f) * (float)crop_w / (float)dst_w - 0.5f;
    float src_y = (y + 0.5f) * (float)crop_h / (float)dst_h - 0.5f;

    int x1 = floorf(src_x);
    int y1 = floorf(src_y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float x_diff = src_x - x1;
    float y_diff = src_y - y1;

    x1 = max(0, x1);
    y1 = max(0, y1);
    x2 = min(crop_w - 1, x2);
    y2 = min(crop_h - 1, y2);

    int abs_x1 = x1 + crop_x;
    int abs_y1 = y1 + crop_y;
    int abs_x2 = x2 + crop_x;
    int abs_y2 = y2 + crop_y;

    for (int c = 0; c < src_c; c++) {
      float p11 = src[(abs_y1 * src_w + abs_x1) * src_c + c];
      float p12 = src[(abs_y2 * src_w + abs_x1) * src_c + c];
      float p21 = src[(abs_y1 * src_w + abs_x2) * src_c + c];
      float p22 = src[(abs_y2 * src_w + abs_x2) * src_c + c];

      float val = p11 * (1.0f - x_diff) * (1.0f - y_diff) +
                  p21 * x_diff * (1.0f - y_diff) +
                  p12 * (1.0f - x_diff) * y_diff + p22 * x_diff * y_diff;

      dst[(y * dst_w + x) * src_c + c] = (val - mean[c]) / std[c];
    }
  }
}

// crop + escale resize + normal
__global__ void escale_resize_normalize_bilinear_kernel(
    const uint8_t *src, float *dst, int src_h, int src_w, int src_c, int dst_h,
    int dst_w, const float *mean, const float *std, const float *pad_val,
    int new_h, int new_w, int pad_y, int pad_x) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dst_w && y < dst_h) {
    if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h) {
      int scaled_x = x - pad_x;
      int scaled_y = y - pad_y;

      float src_x = (scaled_x + 0.5f) * (float)src_w / (float)new_w - 0.5f;
      float src_y = (scaled_y + 0.5f) * (float)src_h / (float)new_h - 0.5f;

      int x1 = floorf(src_x);
      int y1 = floorf(src_y);
      int x2 = x1 + 1;
      int y2 = y1 + 1;

      float x_diff = src_x - x1;
      float y_diff = src_y - y1;

      x1 = max(0, x1);
      y1 = max(0, y1);
      x2 = min(src_w - 1, x2);
      y2 = min(src_h - 1, y2);

      for (int c = 0; c < src_c; c++) {
        float p11 = src[(y1 * src_w + x1) * src_c + c];
        float p12 = src[(y2 * src_w + x1) * src_c + c];
        float p21 = src[(y1 * src_w + x2) * src_c + c];
        float p22 = src[(y2 * src_w + x2) * src_c + c];

        float val = p11 * (1.0f - x_diff) * (1.0f - y_diff) +
                    p21 * x_diff * (1.0f - y_diff) +
                    p12 * (1.0f - x_diff) * y_diff + p22 * x_diff * y_diff;

        dst[(y * dst_w + x) * src_c + c] = (val - mean[c]) / std[c];
      }
    } else {
      for (int c = 0; c < src_c; c++) {
        dst[(y * dst_w + x) * src_c + c] = (pad_val[c] - mean[c]) / std[c];
      }
    }
  }
}
} // namespace kernel

void hwc_to_chw_gpu(const float *src, float *dst, int height, int width,
                    int channels) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  kernel::hwc_to_chw_kernel<<<grid, block>>>(src, dst, height, width, channels);
}

void fp32_to_fp16_gpu(const float *src, uint16_t *dst, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kernel::fp32_to_fp16_clamp_kernel<<<blocks, threads>>>(src, dst, n);
}

void crop_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                               int src_w, int src_c, int crop_x, int crop_y,
                               int crop_h, int crop_w, int dst_h, int dst_w,
                               const float *mean, const float *std) {
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  kernel::crop_resize_normalize_bilinear_kernel<<<grid, block>>>(
      src, dst, src_h, src_w, src_c, crop_x, crop_y, crop_h, crop_w, dst_h,
      dst_w, mean, std);
}

void escale_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                                 int src_w, int src_c, int dst_h, int dst_w,
                                 const float *mean, const float *std,
                                 const float *pad_val) {
  float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
  int new_w = (int)(src_w * scale);
  int new_h = (int)(src_h * scale);
  int pad_w = (dst_w - new_w) / 2;
  int pad_h = (dst_h - new_h) / 2;

  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  kernel::escale_resize_normalize_bilinear_kernel<<<grid, block>>>(
      src, dst, src_h, src_w, src_c, dst_h, dst_w, mean, std, pad_val, new_h,
      new_w, pad_h, pad_w);
}
} // namespace ai_core::dnn::gpu::cuda_op
