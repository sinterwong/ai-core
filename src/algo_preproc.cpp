
#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/tensor_data.hpp"
#include "algo_preproc_impl.hpp"

namespace ai_core::dnn {

AlgoPreproc::AlgoPreproc(const std::string &module_name)
    : m_pImpl(std::make_unique<Impl>(module_name)) {}

AlgoPreproc::~AlgoPreproc() = default;

InferErrorCode AlgoPreproc::initialize(const AlgoPreprocParams &params) {
  return m_pImpl->initialize(params);
}

InferErrorCode
AlgoPreproc::process(const AlgoInput &input, TensorData &model_input,
                     std::shared_ptr<RuntimeContext> &runtime_context,
                     const AlgoPreprocParams *preproc_override) {
  return m_pImpl->process(input, model_input, runtime_context,
                          preproc_override);
}

InferErrorCode
AlgoPreproc::batchProcess(const std::vector<AlgoInput> &input,
                          TensorData &model_input,
                          std::shared_ptr<RuntimeContext> &runtime_context,
                          const AlgoPreprocParams *preproc_override) {
  return m_pImpl->batchProcess(input, model_input, runtime_context,
                               preproc_override);
}

InferErrorCode AlgoPreproc::terminate() { return m_pImpl->terminate(); }

} // namespace ai_core::dnn
