/**
 * @file postproc_registrar.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "postproc_registrar.hpp"
#include "ai_core/ai_core_registrar.hpp"
#include "postproc/anchor_det_postproc.hpp"
#include "postproc/confidence_filter_postproc.hpp"
#include "postproc/cv_generic_postproc.hpp"
#include <logger.hpp>

namespace ai_core::dnn {

DefaultPostprocPluginRegistrar::DefaultPostprocPluginRegistrar() {
  REGISTER_POSTPROCESS_ALGO(AnchorDetPostproc);
  LOG_INFOS << "AnchorDetPostproc registered." << std::endl;

  REGISTER_POSTPROCESS_ALGO(CVGenericPostproc);
  LOG_INFOS << "CVGenericPostproc registered." << std::endl;

  REGISTER_POSTPROCESS_ALGO(ConfidenceFilterPostproc);
  LOG_INFOS << "ConfidenceFilterPostproc registered." << std::endl;
}
} // namespace ai_core::dnn
