/**
 * @file fpr_cls.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_FPR_CLS_HPP_
#define __INFERENCE_VISION_FPR_CLS_HPP_

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"
#include "postproc_base.hpp"
namespace ai_core::dnn {
class FprCls : public PostprocssBase {
public:
  explicit FprCls() {}

  virtual bool process(const TensorData &, AlgoPreprocParams &, AlgoOutput &,
                       AlgoPostprocParams &) override;

  AlgoPostprocParams mParams;
};
} // namespace ai_core::dnn

#endif
