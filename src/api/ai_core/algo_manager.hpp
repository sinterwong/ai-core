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

#include "algo_infer.hpp"
#include "types/infer_error_code.hpp"
#include <memory>
#include <string>

#include "types/algo_data_types.hpp"

namespace ai_core::dnn {

class AlgoManager : public std::enable_shared_from_this<AlgoManager> {
public:
  AlgoManager();
  ~AlgoManager();
  AlgoManager(const AlgoManager &) = delete;
  AlgoManager &operator=(const AlgoManager &) = delete;
  AlgoManager(AlgoManager &&) noexcept;
  AlgoManager &operator=(AlgoManager &&) noexcept;

  InferErrorCode registerAlgo(const std::string &name,
                              const std::shared_ptr<AlgoInference> &algo);

  InferErrorCode unregisterAlgo(const std::string &name);

  InferErrorCode infer(const std::string &name, AlgoInput &input,
                       AlgoPreprocParams &preprocParams, AlgoOutput &output,
                       AlgoPostprocParams &postprocParams);

  std::shared_ptr<AlgoInference> getAlgo(const std::string &name) const;

  bool hasAlgo(const std::string &name) const;

  void clear();

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};

} // namespace ai_core::dnn

#endif // __AI_CORE_ALGO_MANAGER_HPP_