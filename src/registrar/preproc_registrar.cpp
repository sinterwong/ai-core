/**
 * @file preproc_registrar.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "preproc_registrar.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/plugin_registrar.hpp"
#include "preproc/cpu_generic_preprocess.hpp"
#include "preproc/frame_with_mask_prep.hpp"

#ifdef WITH_TRT
#include "preproc/cuda_generic_preprocess.hpp"
#endif

namespace ai_core::dnn {

DefaultPreprocPluginRegistrar::DefaultPreprocPluginRegistrar() {
  REGISTER_PREPROCESS_ALGO(CpuGenericPreprocess);
  LOG_INFO_S << "CpuGenericPreprocess registered." << std::endl;

  REGISTER_PREPROCESS_ALGO(FrameWithMaskPreprocess);
  LOG_INFO_S << "FrameWithMaskPreprocess registered." << std::endl;

#ifdef WITH_TRT
  REGISTER_PREPROCESS_ALGO(CudaGenericPreprocess);
  LOG_INFO_S << "CudaGenericPreprocess registered." << std::endl;
#endif
}
} // namespace ai_core::dnn
