/**
 * @file softmax_cls.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_SOFTMAX_CLS_HPP_
#define __INFERENCE_VISION_SOFTMAX_CLS_HPP_

#include "postproc_base.hpp"
namespace ai_core::dnn {
class SoftmaxCls : public PostprocssBase {
public:
  explicit SoftmaxCls() {}

  virtual bool process(const TensorData &, AlgoPreprocParams &, AlgoOutput &,
                       AlgoPostprocParams &) override;

private:
  AlgoPostprocParams mParams;
};
} // namespace ai_core::dnn

#endif
