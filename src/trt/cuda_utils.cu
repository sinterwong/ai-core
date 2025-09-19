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

namespace ai_core::dnn::gpu::cuda_op
{

  namespace kernel
  {
#define FP16_MAX_VAL 65504.0f

    __global__ void hwc_to_chw_kernel(const float *src, float *dst, int height,
                                      int width, int channels)
    {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x < width && y < height)
      {
        for (int c = 0; c < channels; ++c)
        {
          dst[c * (width * height) + y * width + x] =
              src[y * width * channels + x * channels + c];
        }
      }
    }

    __global__ void fp32_to_fp16_clamp_kernel(const float *src, uint16_t *dst,
                                              int n)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
      {
        // clamp the numerical values to the representational range of fp16
        float val = fminf(fmaxf(src[i], -FP16_MAX_VAL), FP16_MAX_VAL);
        dst[i] = __float2half_rn(val);
      }
    }

    // crop + resize + normal
    __global__ void crop_resize_normalize_bilinear_kernel(
        const uint8_t *src, float *dst, int src_h, int src_w, int src_c, int crop_x,
        int crop_y, int crop_h, int crop_w, int dst_h, int dst_w, const float *mean,
        const float *std)
    {

      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x < dst_w && y < dst_h)
      {
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

        for (int c = 0; c < src_c; c++)
        {
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
        int new_h, int new_w, int pad_y, int pad_x)
    {

      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x < dst_w && y < dst_h)
      {
        if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h)
        {
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

          for (int c = 0; c < src_c; c++)
          {
            float p11 = src[(y1 * src_w + x1) * src_c + c];
            float p12 = src[(y2 * src_w + x1) * src_c + c];
            float p21 = src[(y1 * src_w + x2) * src_c + c];
            float p22 = src[(y2 * src_w + x2) * src_c + c];

            float val = p11 * (1.0f - x_diff) * (1.0f - y_diff) +
                        p21 * x_diff * (1.0f - y_diff) +
                        p12 * (1.0f - x_diff) * y_diff + p22 * x_diff * y_diff;

            dst[(y * dst_w + x) * src_c + c] = (val - mean[c]) / std[c];
          }
        }
        else
        {
          for (int c = 0; c < src_c; c++)
          {
            dst[(y * dst_w + x) * src_c + c] = (pad_val[c] - mean[c]) / std[c];
          }
        }
      }
    }

    __global__ void batch_hwc_to_chw_kernel(const float *src, float *dst, int height,
                                            int width, int channels, int batch_size)
    {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      int b = blockIdx.z; // Batch index

      if (x < width && y < height && b < batch_size)
      {
        size_t image_plane_size = (size_t)height * width;
        size_t image_size = image_plane_size * channels;

        const float *src_img = src + b * image_size;
        float *dst_img = dst + b * image_size;

        for (int c = 0; c < channels; ++c)
        {
          dst_img[c * image_plane_size + y * width + x] =
              src_img[y * width * channels + x * channels + c];
        }
      }
    }

    __global__ void batch_crop_resize_normalize_bilinear_kernel(
        const uint8_t *const *d_src_ptrs, // 指针数组，指向每张图的GPU内存
        float *d_batch_dst,               // 批处理输出缓冲区
        const int *d_src_hs, const int *d_src_ws, int src_c,
        const ROIData *d_rois,
        int dst_h, int dst_w, const float *mean, const float *std, int batch_size)
    {

      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      int b = blockIdx.z; // Batch index

      if (x >= dst_w || y >= dst_h || b >= batch_size)
        return;

      // 获取当前图像的元数据
      const uint8_t *src = d_src_ptrs[b];
      int src_h = d_src_hs[b];
      int src_w = d_src_ws[b];
      ROIData roi = d_rois[b];
      int crop_w = roi.w;
      int crop_h = roi.h;

      // 计算输出在批处理缓冲区中的偏移
      size_t dst_image_size = (size_t)dst_h * dst_w * src_c;
      float *dst = d_batch_dst + b * dst_image_size;

      // 双线性插值逻辑 (与单帧版本相同)
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

      int abs_x1 = x1 + roi.x;
      int abs_y1 = y1 + roi.y;
      int abs_x2 = x2 + roi.x;
      int abs_y2 = y2 + roi.y;

      for (int c = 0; c < src_c; c++)
      {
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

    __global__ void batch_escale_resize_normalize_bilinear_kernel(
        const uint8_t *const *d_src_ptrs, // 指针数组
        float *d_batch_dst,               // 批处理输出
        const int *d_src_hs, const int *d_src_ws, int src_c,
        const ROIData *d_rois,
        int dst_h, int dst_w,
        const float *mean, const float *std, const float *pad_val,
        const int *d_new_hs, const int *d_new_ws, // 每张图缩放后的尺寸
        const int *d_pad_ys, const int *d_pad_xs, // 每张图的padding
        int batch_size)
    {

      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      int b = blockIdx.z; // Batch index

      if (x >= dst_w || y >= dst_h || b >= batch_size)
        return;

      // 获取当前图像的元数据
      const uint8_t *src = d_src_ptrs[b];
      ROIData roi = d_rois[b];
      int src_h = roi.h; // 对于escale, src_h/w 是roi的h/w
      int src_w = roi.w;
      int new_h = d_new_hs[b];
      int new_w = d_new_ws[b];
      int pad_y = d_pad_ys[b];
      int pad_x = d_pad_xs[b];

      // 计算输出在批处理缓冲区中的偏移
      size_t dst_image_size = (size_t)dst_h * dst_w * src_c;
      float *dst = d_batch_dst + b * dst_image_size;

      // 定位到src图像的ROI起点
      const uint8_t *roi_src = src + ((size_t)roi.y * d_src_ws[b] + roi.x) * src_c;
      int src_pitch = d_src_ws[b] * src_c; // 原始图像的行步长

      if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h)
      {
        int scaled_x = x - pad_x;
        int scaled_y = y - pad_y;

        float src_x_f = (scaled_x + 0.5f) * (float)src_w / (float)new_w - 0.5f;
        float src_y_f = (scaled_y + 0.5f) * (float)src_h / (float)new_h - 0.5f;

        int x1 = floorf(src_x_f);
        int y1 = floorf(src_y_f);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        float x_diff = src_x_f - x1;
        float y_diff = src_y_f - y1;

        x1 = max(0, x1);
        y1 = max(0, y1);
        x2 = min(src_w - 1, x2);
        y2 = min(src_h - 1, y2);

        for (int c = 0; c < src_c; c++)
        {
          float p11 = roi_src[(y1 * src_pitch) + x1 * src_c + c];
          float p12 = roi_src[(y2 * src_pitch) + x1 * src_c + c];
          float p21 = roi_src[(y1 * src_pitch) + x2 * src_c + c];
          float p22 = roi_src[(y2 * src_pitch) + x2 * src_c + c];

          float val = p11 * (1.0f - x_diff) * (1.0f - y_diff) +
                      p21 * x_diff * (1.0f - y_diff) +
                      p12 * (1.0f - x_diff) * y_diff + p22 * x_diff * y_diff;

          dst[(y * dst_w + x) * src_c + c] = (val - mean[c]) / std[c];
        }
      }
      else
      {
        for (int c = 0; c < src_c; c++)
        {
          dst[(y * dst_w + x) * src_c + c] = (pad_val[c] - mean[c]) / std[c];
        }
      }
    }

  } // namespace kernel

  void hwc_to_chw_gpu(const float *src, float *dst, int height, int width,
                      int channels)
  {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernel::hwc_to_chw_kernel<<<grid, block>>>(src, dst, height, width, channels);
  }

  void fp32_to_fp16_gpu(const float *src, uint16_t *dst, int n)
  {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel::fp32_to_fp16_clamp_kernel<<<blocks, threads>>>(src, dst, n);
  }

  void crop_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                                 int src_w, int src_c, int crop_x, int crop_y,
                                 int crop_h, int crop_w, int dst_h, int dst_w,
                                 const float *mean, const float *std)
  {
    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    kernel::crop_resize_normalize_bilinear_kernel<<<grid, block>>>(
        src, dst, src_h, src_w, src_c, crop_x, crop_y, crop_h, crop_w, dst_h,
        dst_w, mean, std);
  }

  void escale_resize_normalize_gpu(const uint8_t *src, float *dst, int src_h,
                                   int src_w, int src_c, int dst_h, int dst_w,
                                   const float *mean, const float *std,
                                   const float *pad_val)
  {
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

  void batch_hwc_to_chw_gpu(const float *src, float *dst, int height, int width,
                            int channels, int batch_size)
  {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
    kernel::batch_hwc_to_chw_kernel<<<grid, block>>>(src, dst, height, width, channels, batch_size);
  }

  void batch_crop_resize_normalize_gpu(
      const uint8_t *const *d_src_ptrs, float *d_batch_dst,
      const int *d_src_hs, const int *d_src_ws, int src_c,
      const ROIData *d_rois,
      int dst_h, int dst_w, const float *mean, const float *std, int batch_size)
  {

    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y, batch_size);
    kernel::batch_crop_resize_normalize_bilinear_kernel<<<grid, block>>>(
        d_src_ptrs, d_batch_dst, d_src_hs, d_src_ws, src_c, d_rois,
        dst_h, dst_w, mean, std, batch_size);
  }

  void batch_escale_resize_normalize_gpu(
      const uint8_t *const *d_src_ptrs, float *d_batch_dst,
      const int *d_src_hs, const int *d_src_ws, int src_c,
      const ROIData *d_rois,
      int dst_h, int dst_w, const float *mean, const float *std, const float *pad_val,
      const int *d_new_hs, const int *d_new_ws,
      const int *d_pad_ys, const int *d_pad_xs,
      int batch_size)
  {

    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y, batch_size);
    kernel::batch_escale_resize_normalize_bilinear_kernel<<<grid, block>>>(
        d_src_ptrs, d_batch_dst, d_src_hs, d_src_ws, src_c, d_rois,
        dst_h, dst_w, mean, std, pad_val,
        d_new_hs, d_new_ws, d_pad_ys, d_pad_xs, batch_size);
  }
} // namespace ai_core::dnn::gpu::cuda_op
