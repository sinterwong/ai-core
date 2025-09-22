/**
 * @file preproc_registrar.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_PREPROC_REGISTRAR_HPP
#define AI_CORE_PREPROC_REGISTRAR_HPP

namespace ai_core::dnn {

class DefaultPreprocPluginRegistrar {
public:
  static DefaultPreprocPluginRegistrar &getInstance() {
    static DefaultPreprocPluginRegistrar instance;
    return instance;
  }

  DefaultPreprocPluginRegistrar(const DefaultPreprocPluginRegistrar &) = delete;
  DefaultPreprocPluginRegistrar &
  operator=(const DefaultPreprocPluginRegistrar &) = delete;
  DefaultPreprocPluginRegistrar(DefaultPreprocPluginRegistrar &&) = delete;
  DefaultPreprocPluginRegistrar &
  operator=(DefaultPreprocPluginRegistrar &&) = delete;

private:
  DefaultPreprocPluginRegistrar();
};

[[maybe_unused]] inline const static DefaultPreprocPluginRegistrar
    &__temp_preproc_registrar_ = DefaultPreprocPluginRegistrar::getInstance();
} // namespace ai_core::dnn

#endif