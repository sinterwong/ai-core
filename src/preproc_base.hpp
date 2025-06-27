/**
 * @file preproc_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __PREPROCESS_BASE_HPP_
#define __PREPROCESS_BASE_HPP_

#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/tensor_data.hpp"

namespace ai_core::dnn {

class PreprocssBase {
public:
  explicit PreprocssBase() {}

  virtual ~PreprocssBase(){};

  virtual bool process(AlgoInput &, AlgoPreprocParams &params,
                       TensorData &) = 0;
};
} // namespace ai_core::dnn

#endif