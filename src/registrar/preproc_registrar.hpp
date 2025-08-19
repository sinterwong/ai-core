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
#ifndef __PREPROC_REGISTRAR_HPP__
#define __PREPROC_REGISTRAR_HPP__

#include "preproc_base.hpp"
#include "type_safe_factory.hpp"

namespace ai_core::dnn {

using PreprocFactory = Factory<PreprocssBase>;

class PreprocessRegistrar {
public:
  static PreprocessRegistrar &getInstance() {
    static PreprocessRegistrar instance;
    return instance;
  }

  PreprocessRegistrar(const PreprocessRegistrar &) = delete;
  PreprocessRegistrar &operator=(const PreprocessRegistrar &) = delete;
  PreprocessRegistrar(PreprocessRegistrar &&) = delete;
  PreprocessRegistrar &operator=(PreprocessRegistrar &&) = delete;

private:
  PreprocessRegistrar();
};

[[maybe_unused]] inline const static PreprocessRegistrar &preproc_registrar =
    PreprocessRegistrar::getInstance();
} // namespace ai_core::dnn

#endif