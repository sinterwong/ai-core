/**
 * @file gpu_generic_cuda_preprocessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __GPU_GENERIC_CUDA_PREPROCESSOR_HPP__
#define __GPU_GENERIC_CUDA_PREPROCESSOR_HPP__

#include "ai_core/algo_input_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "preproc/frame_preprocessor_base.hpp"

namespace ai_core::dnn::gpu {

class GpuGenericCudaPreprocessor : public IFramePreprocessor {
public:
  explicit GpuGenericCudaPreprocessor() = default;
  ~GpuGenericCudaPreprocessor() override = default;

  TypedBuffer process(FramePreprocessArg &args,
                      const FrameInput &input) const override;
};

} // namespace ai_core::dnn::gpu

#endif // __GPU_GENERIC_CUDA_PREPROCESSOR_HPP__
