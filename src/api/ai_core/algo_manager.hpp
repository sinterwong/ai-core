/**
 * @file vision_algo_manager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __AI_CORE_ALGO_MANAGER_HPP_
#define __AI_CORE_ALGO_MANAGER_HPP_

#include "algo_infer_base.hpp" // For std::shared_ptr<AlgoInferBase>
#include "types/infer_error_code.hpp" // For InferErrorCode
#include <memory> // For std::unique_ptr, std::shared_ptr, std::enable_shared_from_this
#include <string> // For std::string

// Forward declare types used in method signatures if their full definition isn't needed
// However, AlgoInput& and AlgoOutput& will need full definitions.
#include "types/algo_data_types.hpp" // For AlgoInput, AlgoOutput

namespace ai_core::dnn {

class AlgoManager : public std::enable_shared_from_this<AlgoManager> {
public:
  AlgoManager(); // Constructor needs to be defined in .cpp
  ~AlgoManager(); // Destructor needs to be defined in .cpp for unique_ptr to incomplete type

  AlgoManager(const AlgoManager &) = delete;
  AlgoManager &operator=(const AlgoManager &) = delete;
  AlgoManager(AlgoManager &&) noexcept;            // Optional: move constructor
  AlgoManager &operator=(AlgoManager &&) noexcept; // Optional: move assignment

  InferErrorCode registerAlgo(const std::string &name,
                              const std::shared_ptr<AlgoInferBase> &algo);

  InferErrorCode unregisterAlgo(const std::string &name);

  InferErrorCode infer(const std::string &name, AlgoInput &input,
                       AlgoOutput &output);

  std::shared_ptr<AlgoInferBase> getAlgo(const std::string &name) const;

  bool hasAlgo(const std::string &name) const;

  void clear();

private:
  // Forward declaration of the implementation class
  class Impl;
  std::unique_ptr<Impl> pImpl;
};

} // namespace ai_core::dnn

#endif // __AI_CORE_ALGO_MANAGER_HPP_