/**
 * @file algo_manager_impl.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Implementation for AlgoManager PImpl
 * @version 0.1
 * @date 2025-07-15 (Update date)
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "algo_manager_impl.hpp"
#include "logger.hpp"

namespace ai_core::dnn {

AlgoManager::Impl::Impl() {}

AlgoManager::Impl::~Impl() {}

InferErrorCode
AlgoManager::Impl::registerAlgo(const std::string &name,
                                const std::shared_ptr<AlgoInference> &algo) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (algoMap_.count(name)) {
    LOG_ERRORS << "Algo with name " << name << " already registered.";
    return InferErrorCode::ALGO_REGISTER_FAILED;
  }
  if (!algo) {
    LOG_ERRORS << "Attempted to register a null algo with name " << name;
    return InferErrorCode::ALGO_REGISTER_FAILED;
  }
  algoMap_[name] = algo;
  LOG_INFOS << "Registered algo: " << name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoManager::Impl::unregisterAlgo(const std::string &name) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (algoMap_.count(name)) {
    algoMap_.erase(name);
    LOG_INFOS << "Unregistered algo: " << name;
  } else {
    LOG_WARNINGS << "Attempted to unregister non-existent algo: " << name;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoManager::Impl::infer(const std::string &name,
                                        AlgoInput &input,
                                        AlgoPreprocParams &preprocParams,
                                        AlgoOutput &output,
                                        AlgoPostprocParams &postprocParams) {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = algoMap_.find(name);
  if (it == algoMap_.end()) {
    LOG_ERRORS << "Algo with name " << name << " not found for inference.";
    return InferErrorCode::ALGO_NOT_FOUND;
  }
  if (!it->second) {
    LOG_ERRORS << "Algo with name " << name << " is registered but null.";
    return InferErrorCode::ALGO_NOT_FOUND;
  }
  return it->second->infer(input, preprocParams, output, postprocParams);
}

std::shared_ptr<AlgoInference>
AlgoManager::Impl::getAlgo(const std::string &name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = algoMap_.find(name);
  if (it == algoMap_.end()) {
    LOG_ERRORS << "Algo with name " << name << " not found in getAlgo.";
    return nullptr;
  }
  return it->second;
}

bool AlgoManager::Impl::hasAlgo(const std::string &name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return algoMap_.count(name) > 0;
}

void AlgoManager::Impl::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  algoMap_.clear();
  LOG_INFOS << "Cleared all registered algos.";
}

} // namespace ai_core::dnn
