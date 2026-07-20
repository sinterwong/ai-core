/**
 * @file algo_infer_engine.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_ALGO_INFER_ENGINE_HPP
#define AI_CORE_ALGO_INFER_ENGINE_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

/**
 * @brief Thin wrapper around a backend inference engine plugin.
 *
 * @par Thread safety
 * Concurrency-safe per instance for @ref infer (the backend serializes: ORT
 * concurrent via shared lock, NCNN/TRT via a mutex). @ref initialize and
 * @ref terminate require exclusive access. Distinct instances are independent.
 */
class AlgoInferEngine {
public:
  AlgoInferEngine(const std::string &module_name,
                  const AlgoInferParams &infer_params);

  ~AlgoInferEngine();

  InferErrorCode initialize();

  InferErrorCode infer(const TensorData &model_input, TensorData &model_output);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
