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

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

class PreprocssBase {
public:
  virtual ~PreprocssBase(){};

  virtual bool process(const AlgoInput &, const AlgoPreprocParams &,
                       TensorData &,
                       std::shared_ptr<RuntimeContext> &) const = 0;
};
} // namespace ai_core::dnn

#endif