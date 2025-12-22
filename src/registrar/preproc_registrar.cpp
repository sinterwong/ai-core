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
#include "ai_core/ai_core_registrar.hpp"
#include "ai_core/logger.hpp"
#include "preproc/frame_prep.hpp"
#include "preproc/frame_with_mask_prep.hpp"

namespace ai_core::dnn {

DefaultPreprocPluginRegistrar::DefaultPreprocPluginRegistrar() {
  REGISTER_PREPROCESS_ALGO(FramePreprocess);
  LOG_INFO_S << "FramePreprocess registered." << std::endl;

  REGISTER_PREPROCESS_ALGO(FrameWithMaskPreprocess);
  LOG_INFO_S << "FrameWithMaskPreprocess registered." << std::endl;
}
} // namespace ai_core::dnn
