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

#include "ai_core/algo_manager.hpp" // Public header
#include "algo_manager_impl.hpp"    // Private implementation header

// Note: AlgoInput, AlgoOutput, InferErrorCode etc. are pulled in via algo_manager.hpp
// which includes the necessary public type headers from ai_core/types/.
// AlgoInferBase is also included via algo_manager.hpp.

namespace ai_core::dnn {

// Constructor
AlgoManager::AlgoManager() : pImpl(std::make_unique<Impl>()) {}

// Destructor
// Required for std::unique_ptr to an incomplete type in the header.
// The compiler needs to see the full definition of Impl when generating
// the destructor code for unique_ptr.
AlgoManager::~AlgoManager() = default; // Default is fine as long as Impl's destructor is accessible

// Move constructor
// The default move constructor will correctly move the std::unique_ptr pImpl.
AlgoManager::AlgoManager(AlgoManager &&other) noexcept = default;

// Move assignment operator
// The default move assignment operator will correctly move assign the std::unique_ptr pImpl.
AlgoManager &AlgoManager::operator=(AlgoManager &&other) noexcept = default;


InferErrorCode
AlgoManager::registerAlgo(const std::string &name,
                          const std::shared_ptr<AlgoInferBase> &algo) {
  return pImpl->registerAlgo(name, algo);
}

InferErrorCode AlgoManager::unregisterAlgo(const std::string &name) {
  return pImpl->unregisterAlgo(name);
}

InferErrorCode AlgoManager::infer(const std::string &name, AlgoInput &input,
                                  AlgoOutput &output) {
  return pImpl->infer(name, input, output);
}

std::shared_ptr<AlgoInferBase>
AlgoManager::getAlgo(const std::string &name) const {
  return pImpl->getAlgo(name);
}

bool AlgoManager::hasAlgo(const std::string &name) const {
  return pImpl->hasAlgo(name);
}

void AlgoManager::clear() { pImpl->clear(); }

} // namespace ai_core::dnn