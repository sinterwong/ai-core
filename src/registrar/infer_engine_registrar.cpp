/**
 * @file algo_registrar.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "infer_engine_registrar.hpp"
#include "ai_core/types/algo_data_types.hpp"
#include "logger.hpp"

#ifdef WITH_ORT
#include "ort/dnn_infer.hpp"
#endif

namespace ai_core::dnn {

#define REGISTER_INFER_ENGINE(EngineName)                                      \
  InferEngineFactory::instance().registerCreator(                              \
      #EngineName,                                                             \
      [](const AlgoConstructParams &cparams) -> std::shared_ptr<InferBase> {   \
        return std::make_shared<EngineName>(cparams);                          \
      });                                                                      \
  LOG_INFOS << "Registered " #EngineName " creator."

InferEngineRegistrar::InferEngineRegistrar() {
#ifdef WITH_ORT
  REGISTER_INFER_ENGINE(OrtAlgoInference);
#endif
}
} // namespace ai_core::dnn
