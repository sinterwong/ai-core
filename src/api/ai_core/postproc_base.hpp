/**
 * @file postproc_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_POSTPROC_BASE_HPP
#define AI_CORE_POSTPROC_BASE_HPP

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

class PostprocssBase {
public:
  virtual ~PostprocssBase(){};

  virtual bool process(const TensorData &, const AlgoPostprocParams &,
                       AlgoOutput &,
                       std::shared_ptr<RuntimeContext> &) const = 0;

  virtual bool batchProcess(const TensorData &, const AlgoPostprocParams &,
                            std::vector<AlgoOutput> &,
                            std::shared_ptr<RuntimeContext> &) const = 0;
};
} // namespace ai_core::dnn

#endif