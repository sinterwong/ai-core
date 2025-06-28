/**
 * @file algo_manager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Manages algorithm instances and inference execution.
 * @version 0.2 (Refactored with PImpl)
 * @date 2025-07-15 (Update date)
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ai_core/algo_manager.hpp"
#include "algo_manager_impl.hpp"

namespace ai_core::dnn {

AlgoManager::AlgoManager() : pImpl(std::make_unique<Impl>()) {}

AlgoManager::~AlgoManager() = default;

AlgoManager::AlgoManager(AlgoManager &&other) noexcept = default;

AlgoManager &AlgoManager::operator=(AlgoManager &&other) noexcept = default;

InferErrorCode
AlgoManager::registerAlgo(const std::string &name,
                          const std::shared_ptr<AlgoInference> &algo) {
  return pImpl->registerAlgo(name, algo);
}

InferErrorCode AlgoManager::unregisterAlgo(const std::string &name) {
  return pImpl->unregisterAlgo(name);
}

InferErrorCode AlgoManager::infer(const std::string &name, AlgoInput &input,
                                  AlgoPreprocParams &preprocParams,
                                  AlgoOutput &output,
                                  AlgoPostprocParams &postprocParams) {
  return pImpl->infer(name, input, preprocParams, output, postprocParams);
}

std::shared_ptr<AlgoInference>
AlgoManager::getAlgo(const std::string &name) const {
  return pImpl->getAlgo(name);
}

bool AlgoManager::hasAlgo(const std::string &name) const {
  return pImpl->hasAlgo(name);
}

void AlgoManager::clear() { pImpl->clear(); }

} // namespace ai_core::dnn