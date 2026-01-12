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
#include "ai_core/logger.hpp"
#include <ostream>

namespace ai_core::dnn {

AlgoManager::Impl::Impl() {}

AlgoManager::Impl::~Impl() {}

InferErrorCode
AlgoManager::Impl::registerAlgo(const std::string &name,
                                const std::shared_ptr<AlgoInference> &algo) {
  std::unique_lock<std::shared_mutex> lock(m_mutex);
  if (m_algoMap.count(name)) {
    LOG_ERROR_S << "Algo with name " << name << " already registered.";
    return InferErrorCode::AlgoRegisterFailed;
  }
  if (!algo) {
    LOG_ERROR_S << "Attempted to register a null algo with name " << name;
    return InferErrorCode::AlgoRegisterFailed;
  }
  m_algoMap[name] = algo;
  LOG_INFO_S << "Registered algo: " << name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoManager::Impl::unregisterAlgo(const std::string &name) {
  std::unique_lock<std::shared_mutex> lock(m_mutex);
  if (m_algoMap.count(name)) {
    m_algoMap.erase(name);
    LOG_INFO_S << "Unregistered algo: " << name;
  } else {
    LOG_WARNING_S << "Attempted to unregister non-existent algo: " << name;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoManager::Impl::infer(const std::string &name,
                                        AlgoInput &input,
                                        AlgoPreprocParams &preproc_params,
                                        AlgoOutput &output,
                                        AlgoPostprocParams &postproc_params) {
  std::shared_lock<std::shared_mutex> lock(m_mutex);
  auto it = m_algoMap.find(name);
  if (it == m_algoMap.end()) {
    LOG_ERROR_S << "Algo with name " << name << " not found for inference.";
    return InferErrorCode::AlgoNotFound;
  }
  if (!it->second) {
    LOG_ERROR_S << "Algo with name " << name << " is registered but null.";
    return InferErrorCode::AlgoNotFound;
  }
  return it->second->infer(input, preproc_params, postproc_params, output);
}

std::shared_ptr<AlgoInference>
AlgoManager::Impl::getAlgo(const std::string &name) const {
  std::shared_lock<std::shared_mutex> lock(m_mutex);
  auto it = m_algoMap.find(name);
  if (it == m_algoMap.end()) {
    LOG_ERROR_S << "Algo with name " << name << " not found in getAlgo.";
    return nullptr;
  }
  return it->second;
}

bool AlgoManager::Impl::hasAlgo(const std::string &name) const {
  std::shared_lock<std::shared_mutex> lock(m_mutex);
  return m_algoMap.count(name) > 0;
}

void AlgoManager::Impl::clear() {
  std::unique_lock<std::shared_mutex> lock(m_mutex);
  m_algoMap.clear();
  LOG_INFO_S << "Cleared all registered algos.";
}

} // namespace ai_core::dnn
