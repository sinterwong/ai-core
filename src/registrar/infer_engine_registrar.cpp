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
#include "ai_core/ai_core_registrar.hpp"
#include "ai_core/logger.hpp"

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

DefaultInferEnginePluginRegistrar::DefaultInferEnginePluginRegistrar() {
#ifdef WITH_ORT
  REGISTER_INFER_ENGINE(OrtAlgoInference);
  LOG_INFO_S << "ONNXRuntime inference engine registered." << std::endl;
#endif

#ifdef WITH_NCNN
  REGISTER_INFER_ENGINE(NCNNAlgoInference);
  LOG_INFO_S << "NCNN inference engine registered." << std::endl;
#endif

#ifdef WITH_TRT
  REGISTER_INFER_ENGINE(TrtAlgoInference);
  LOG_INFO_S << "TensorRT inference engine registered." << std::endl;
#endif
}
} // namespace ai_core::dnn
