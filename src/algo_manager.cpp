
#include "ai_core/algo_manager.hpp"
#include "algo_manager_impl.hpp"

namespace ai_core::dnn {

AlgoManager::AlgoManager() : m_pImpl(std::make_unique<Impl>()) {}

AlgoManager::~AlgoManager() = default;

AlgoManager::AlgoManager(AlgoManager &&other) noexcept = default;

AlgoManager &AlgoManager::operator=(AlgoManager &&other) noexcept = default;

InferErrorCode
AlgoManager::registerAlgo(const std::string &name,
                          const std::shared_ptr<AlgoInference> &algo) {
  return m_pImpl->registerAlgo(name, algo);
}

InferErrorCode AlgoManager::unregisterAlgo(const std::string &name) {
  return m_pImpl->unregisterAlgo(name);
}

InferErrorCode AlgoManager::infer(const std::string &name, AlgoInput &input,
                                  AlgoOutput &output) {
  return m_pImpl->infer(name, input, output);
}

std::shared_ptr<AlgoInference>
AlgoManager::getAlgo(const std::string &name) const {
  return m_pImpl->getAlgo(name);
}

bool AlgoManager::hasAlgo(const std::string &name) const {
  return m_pImpl->hasAlgo(name);
}

void AlgoManager::clear() { m_pImpl->clear(); }

} // namespace ai_core::dnn
