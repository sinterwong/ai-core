/**
 * @file plugin_registrar.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-09-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_REGISTRAR_HPP
#define AI_CORE_REGISTRAR_HPP
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/type_safe_factory.hpp"

namespace ai_core::dnn {

using PreprocFactory = Factory<IPreprocessPlugin>;

using InferEngineFactory = Factory<IInferEnginePlugin>;

using PostprocFactory = Factory<IPostprocessPlugin>;

#define REGISTER_PREPROCESS_ALGO(AlgoName)                                     \
  PreprocFactory::instance().registerCreator(                                  \
      #AlgoName,                                                               \
      [](const AlgoConstructParams &cparams)                                   \
          -> std::shared_ptr<IPreprocessPlugin> {                              \
        return std::make_shared<AlgoName>();                                   \
      });

#define REGISTER_INFER_ENGINE(EngineName)                                      \
  InferEngineFactory::instance().registerCreator(                              \
      #EngineName,                                                             \
      [](const AlgoConstructParams &cparams)                                   \
          -> std::shared_ptr<IInferEnginePlugin> {                             \
        return std::make_shared<EngineName>(cparams);                          \
      });

#define REGISTER_POSTPROCESS_ALGO(AlgoName)                                    \
  PostprocFactory::instance().registerCreator(                                 \
      #AlgoName,                                                               \
      [](const AlgoConstructParams &cparams)                                   \
          -> std::shared_ptr<IPostprocessPlugin> {                             \
        return std::make_shared<AlgoName>();                                   \
      });

} // namespace ai_core::dnn
#endif