/**
 * @file algo_manager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-09
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "algo_manager.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"

namespace infer::dnn {

InferErrorCode
AlgoManager::registerAlgo(const std::string &name,
                          const std::shared_ptr<AlgoInferBase> &algo) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (algoMap_.count(name)) {
    LOG_ERRORS << "Algo with name " << name << " already registered.";
    return InferErrorCode::ALGO_REGISTER_FAILED;
  }
  algoMap_[name] = algo;
  LOG_INFOS << "Registered algo: " << name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoManager::unregisterAlgo(const std::string &name) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (algoMap_.count(name)) {
    algoMap_.erase(name);
    LOG_INFOS << "Unregistered algo: " << name;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoManager::infer(const std::string &name, AlgoInput &input,
                                  AlgoOutput &output) {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = algoMap_.find(name);
  if (it == algoMap_.end()) {
    LOG_ERRORS << "Algo with name " << name << " not found.";
    return InferErrorCode::ALGO_INFER_FAILED;
  }
  return it->second->infer(input, output);
}

std::shared_ptr<AlgoInferBase>
AlgoManager::getAlgo(const std::string &name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = algoMap_.find(name);
  if (it == algoMap_.end()) {
    LOG_ERRORS << "Algo with name " << name << " not found.";
    return nullptr;
  }
  return it->second;
}

bool AlgoManager::hasAlgo(const std::string &name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return algoMap_.count(name) > 0;
}

void AlgoManager::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  algoMap_.clear();
  LOG_INFOS << "Cleared all registered algos.";
}

} // namespace infer::dnn