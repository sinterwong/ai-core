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
#include "postproc/anchor_det_postproc.hpp"
#include "postproc/confidence_filter_postproc.hpp"
#include "postproc/cv_generic_postproc.hpp"
#include "postproc_base.hpp"
#include "type_safe_factory.hpp"
#include <logger.hpp>
#include <ostream>

namespace ai_core::dnn {
#define REGISTER_POSTPROCESS_ALGO(AlgoName)                                    \
  PostprocFactory::instance().registerCreator(                                 \
      #AlgoName,                                                               \
      [](const AlgoConstructParams &cparams)                                   \
          -> std::shared_ptr<PostprocssBase> {                                 \
        return std::make_shared<AlgoName>();                                   \
      });                                                                      \
  LOG_INFOS << "Registered " #AlgoName " creator."

PostprocessRegistrar::PostprocessRegistrar() {
  REGISTER_POSTPROCESS_ALGO(AnchorDetPostproc);
  REGISTER_POSTPROCESS_ALGO(CVGenericPostproc);
  REGISTER_POSTPROCESS_ALGO(ConfidenceFilterPostproc);
}
} // namespace ai_core::dnn
