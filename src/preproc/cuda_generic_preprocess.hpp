/**
 * @file cuda_generic_preprocess.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Generic frame preprocessing plugin backed by the CUDA kernel
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_CUDA_GENERIC_PREPROCESS_HPP
#define AI_CORE_CUDA_GENERIC_PREPROCESS_HPP

#ifdef WITH_TRT

#include "generic_frame_preproc_base.hpp"
#include "gpu_generic_cuda_preprocessor.hpp"

namespace ai_core::dnn {

class CudaGenericPreprocess final : public GenericFramePreprocBase {
protected:
  const IFramePreprocessor &kernel() const override { return m_kernel; }

private:
  gpu::GpuGenericCudaPreprocessor m_kernel;
};

} // namespace ai_core::dnn

#endif // WITH_TRT

#endif
