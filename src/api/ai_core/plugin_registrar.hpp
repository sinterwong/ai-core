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
#include <memory>

namespace ai_core::dnn {

using PreprocFactory = Factory<IPreprocessPlugin>;

using InferEngineFactory = Factory<IInferEnginePlugin>;

using PostprocFactory = Factory<IPostprocessPlugin>;

// The macro bodies below fully qualify every ai_core name. These are public
// macros for out-of-tree plugins, so they must expand correctly from any
// namespace — an unqualified `AlgoConstructParams` would force the caller to
// have both `ai_core` and `ai_core::dnn` in scope. Only `AlgoName` is left
// unqualified: it is the caller's own type, resolved at the call site.

#define REGISTER_PREPROCESS_ALGO(AlgoName)                                     \
  ::ai_core::dnn::PreprocFactory::instance().registerCreator(                  \
      #AlgoName,                                                               \
      [](const ::ai_core::AlgoConstructParams &cparams)                        \
          -> std::shared_ptr<::ai_core::dnn::IPreprocessPlugin> {              \
        return std::make_shared<AlgoName>();                                   \
      });

#define REGISTER_INFER_ENGINE(EngineName)                                      \
  ::ai_core::dnn::InferEngineFactory::instance().registerCreator(              \
      #EngineName,                                                             \
      [](const ::ai_core::AlgoConstructParams &cparams)                        \
          -> std::shared_ptr<::ai_core::dnn::IInferEnginePlugin> {             \
        return std::make_shared<EngineName>(cparams);                          \
      });

#define REGISTER_POSTPROCESS_ALGO(AlgoName)                                    \
  ::ai_core::dnn::PostprocFactory::instance().registerCreator(                 \
      #AlgoName,                                                               \
      [](const ::ai_core::AlgoConstructParams &cparams)                        \
          -> std::shared_ptr<::ai_core::dnn::IPostprocessPlugin> {             \
        return std::make_shared<AlgoName>();                                   \
      });

} // namespace ai_core::dnn
#endif