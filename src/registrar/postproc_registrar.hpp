/**
 * @file postproc_registrar.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_POSTPROC_REGISTRAR_HPP
#define AI_CORE_POSTPROC_REGISTRAR_HPP

namespace ai_core::dnn {
class DefaultPostprocPluginRegistrar {
public:
  static DefaultPostprocPluginRegistrar &getInstance() {
    static DefaultPostprocPluginRegistrar instance;
    return instance;
  }

  DefaultPostprocPluginRegistrar(const DefaultPostprocPluginRegistrar &) =
      delete;
  DefaultPostprocPluginRegistrar &
  operator=(const DefaultPostprocPluginRegistrar &) = delete;
  DefaultPostprocPluginRegistrar(DefaultPostprocPluginRegistrar &&) = delete;
  DefaultPostprocPluginRegistrar &
  operator=(DefaultPostprocPluginRegistrar &&) = delete;

private:
  DefaultPostprocPluginRegistrar();
};

[[maybe_unused]] inline const static DefaultPostprocPluginRegistrar
    &__temp_postproc_registrar_ = DefaultPostprocPluginRegistrar::getInstance();
} // namespace ai_core::dnn

#endif