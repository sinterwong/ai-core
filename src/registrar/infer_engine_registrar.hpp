/**
 * @file infer_engine_registrar.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFER_ENGINE_REGISTRAR_HPP
#define AI_CORE_INFER_ENGINE_REGISTRAR_HPP

namespace ai_core::dnn {

class DefaultInferEnginePluginRegistrar {
public:
  static DefaultInferEnginePluginRegistrar &getInstance() {
    static DefaultInferEnginePluginRegistrar instance;
    return instance;
  }

  DefaultInferEnginePluginRegistrar(const DefaultInferEnginePluginRegistrar &) =
      delete;
  DefaultInferEnginePluginRegistrar &
  operator=(const DefaultInferEnginePluginRegistrar &) = delete;
  DefaultInferEnginePluginRegistrar(DefaultInferEnginePluginRegistrar &&) =
      delete;
  DefaultInferEnginePluginRegistrar &
  operator=(DefaultInferEnginePluginRegistrar &&) = delete;

private:
  DefaultInferEnginePluginRegistrar();
};

[[maybe_unused]] inline const static DefaultInferEnginePluginRegistrar
    &__temp_infer_engine_registrar_ =
        DefaultInferEnginePluginRegistrar::getInstance();

} // namespace ai_core::dnn

#endif