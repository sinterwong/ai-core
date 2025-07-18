/**
 * @file algo_postproc.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ai_core/algo_postproc.hpp"
#include "ai_core/tensor_data.hpp"
#include "algo_postproc_impl.hpp"

namespace ai_core::dnn {

AlgoPostproc::AlgoPostproc(const std::string &moduleName,
                           const AlgoPostprocParams &postprocParams)
    : pImpl(std::make_unique<Impl>(moduleName, postprocParams)) {}

AlgoPostproc::~AlgoPostproc() = default;

InferErrorCode AlgoPostproc::initialize() { return pImpl->initialize(); }

InferErrorCode AlgoPostproc::process(const TensorData &modelOutput,
                                     AlgoPreprocParams &preprocParams,
                                     AlgoOutput &output,
                                     AlgoPostprocParams &postprocParams) {
  return pImpl->process(modelOutput, preprocParams, output, postprocParams);
}

InferErrorCode AlgoPostproc::terminate() { return pImpl->terminate(); }

} // namespace ai_core::dnn
