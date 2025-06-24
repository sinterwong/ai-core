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
#ifndef __CORE_ALGO_MANAGER_HPP_
#define __CORE_ALGO_MANAGER_HPP_
#include "algo_infer_base.hpp"
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace infer::dnn {
class AlgoManager : public std::enable_shared_from_this<AlgoManager> {
public:
  AlgoManager() = default;
  ~AlgoManager() = default;

  InferErrorCode registerAlgo(const std::string &name,
                              const std::shared_ptr<AlgoInferBase> &algo);

  InferErrorCode unregisterAlgo(const std::string &name);

  InferErrorCode infer(const std::string &name, AlgoInput &input,
                       AlgoOutput &output);

  std::shared_ptr<AlgoInferBase> getAlgo(const std::string &name) const;

  bool hasAlgo(const std::string &name) const;

  void clear();

private:
  std::unordered_map<std::string, std::shared_ptr<AlgoInferBase>> algoMap_;
  mutable std::shared_mutex mutex_;
};

} // namespace infer::dnn

#endif