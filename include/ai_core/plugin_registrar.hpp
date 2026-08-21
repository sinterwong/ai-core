
#ifndef AI_CORE_REGISTRAR_HPP
#define AI_CORE_REGISTRAR_HPP
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/plugin_registry.hpp"
#include <memory>

namespace ai_core::dnn {

// Public registration macros may be expanded in any namespace. Framework
// identifiers are therefore fully qualified; only the caller-owned plugin type
// is intentionally resolved at the expansion site.

/** Register a default-constructible preprocessing plugin by its type name. */
#define REGISTER_PREPROCESS_ALGO(AlgoName)                                     \
  ::ai_core::dnn::PluginRegistry::instance().registerPreprocessor(             \
      #AlgoName,                                                               \
      [](const ::ai_core::AlgoConstructParams &cparams)                        \
          -> std::shared_ptr<::ai_core::dnn::IPreprocessPlugin> {              \
        return std::make_shared<AlgoName>();                                   \
      });

/** Register an inference plugin constructed from `AlgoConstructParams`. */
#define REGISTER_INFER_ENGINE(EngineName)                                      \
  ::ai_core::dnn::PluginRegistry::instance().registerInferenceEngine(          \
      #EngineName,                                                             \
      [](const ::ai_core::AlgoConstructParams &cparams)                        \
          -> std::shared_ptr<::ai_core::dnn::IInferEnginePlugin> {             \
        return std::make_shared<EngineName>(cparams);                          \
      });

/** Register a default-constructible postprocessing plugin by its type name. */
#define REGISTER_POSTPROCESS_ALGO(AlgoName)                                    \
  ::ai_core::dnn::PluginRegistry::instance().registerPostprocessor(            \
      #AlgoName,                                                               \
      [](const ::ai_core::AlgoConstructParams &cparams)                        \
          -> std::shared_ptr<::ai_core::dnn::IPostprocessPlugin> {             \
        return std::make_shared<AlgoName>();                                   \
      });

} // namespace ai_core::dnn
#endif
