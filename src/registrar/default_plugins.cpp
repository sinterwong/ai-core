/**
 * @file default_plugins.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "ai_core/default_plugins.hpp"
#include "infer_engine_registrar.hpp"
#include "postproc_registrar.hpp"
#include "preproc_registrar.hpp"

namespace ai_core::dnn {

void registerDefaultPlugins() {
  // Meyers singletons: each constructor runs exactly once, on first call.
  DefaultPreprocPluginRegistrar::getInstance();
  DefaultInferEnginePluginRegistrar::getInstance();
  DefaultPostprocPluginRegistrar::getInstance();
}

} // namespace ai_core::dnn
