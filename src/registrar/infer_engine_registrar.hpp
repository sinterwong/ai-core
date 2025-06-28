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
#ifndef __INFER_ENGINE_REGISTRAR_HPP__
#define __INFER_ENGINE_REGISTRAR_HPP__

#include "infer_base.hpp"
#include "type_safe_factory.hpp"

namespace ai_core::dnn {

using InferEngineFactory = common_utils::Factory<InferBase>;

class InferEngineRegistrar {
public:
  static InferEngineRegistrar &getInstance() {
    static InferEngineRegistrar instance;
    return instance;
  }

  InferEngineRegistrar(const InferEngineRegistrar &) = delete;
  InferEngineRegistrar &operator=(const InferEngineRegistrar &) = delete;
  InferEngineRegistrar(InferEngineRegistrar &&) = delete;
  InferEngineRegistrar &operator=(InferEngineRegistrar &&) = delete;

private:
  InferEngineRegistrar();
};

[[maybe_unused]] inline const static InferEngineRegistrar
    &infer_engine_registrar = InferEngineRegistrar::getInstance();

} // namespace ai_core::dnn

#endif