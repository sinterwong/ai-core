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
#ifndef __POSTPROCESS_BASE_HPP_
#define __POSTPROCESS_BASE_HPP_

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

class PostprocssBase {
public:
  explicit PostprocssBase() {}

  virtual ~PostprocssBase(){};

  virtual bool process(const TensorData &, AlgoPreprocParams &, AlgoOutput &,
                       AlgoPostprocParams &) = 0;
};
} // namespace ai_core::dnn

#endif