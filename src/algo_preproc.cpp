/**
 * @file algo_preproc.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ai_core/algo_preproc.hpp"
#include "ai_core/tensor_data.hpp"
#include "algo_preproc_impl.hpp"

namespace ai_core::dnn {

AlgoPreproc::AlgoPreproc(const std::string &moduleName)
    : pImpl(std::make_unique<Impl>(moduleName)) {}

AlgoPreproc::~AlgoPreproc() = default;

InferErrorCode AlgoPreproc::initialize() { return pImpl->initialize(); }

InferErrorCode AlgoPreproc::process(AlgoInput &input,
                                    AlgoPreprocParams &preprocParams,
                                    TensorData &modelInput) {
  return pImpl->process(input, preprocParams, modelInput);
}

InferErrorCode AlgoPreproc::terminate() { return pImpl->terminate(); }

} // namespace ai_core::dnn
