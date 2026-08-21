#ifndef ALGO_INFER_ENGINE_IMPL_HPP
#define ALGO_INFER_ENGINE_IMPL_HPP

#include "ai_core/i_infer_engine.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/infer_engine_wrapper.hpp"
#include <memory>
#include <string>

namespace ai_core::dnn {

class AlgoInferEngine::Impl {
public:
  Impl(const std::string &module_name, const AlgoInferParams &infer_params);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode infer(const TensorData &model_input, TensorData &model_output);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

  std::shared_ptr<IAsyncInferEngine> getAsyncEngine() const noexcept;

private:
  std::string m_moduleName;
  AlgoInferParams m_inferParams;
  std::shared_ptr<IInferEnginePlugin> m_engine;
};
} // namespace ai_core::dnn
#endif
