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
#include <ostream>

#include "ai_core/algo_data_types.hpp"
#include "dnn_infer.hpp"
#include "infer_engine_registrar.hpp"

#include <logger.hpp>

#ifdef WITH_ORT
#include "ort/dnn_infer.hpp"
#endif

#ifdef WITH_NCNN
#include "ncnn/dnn_infer.hpp"
#endif

#ifdef WITH_TRT
#include "trt/dnn_infer.hpp"
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

#ifdef WITH_NCNN
  REGISTER_INFER_ENGINE(NCNNAlgoInference);
#endif

#ifdef WITH_TRT
  REGISTER_INFER_ENGINE(TrtAlgoInference);
#endif
}
} // namespace ai_core::dnn
