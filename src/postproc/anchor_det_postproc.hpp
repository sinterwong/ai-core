/**
 * @file anchor_det_postproc.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __ANCHOR_DET_POSTPROCESSOR_HPP__
#define __ANCHOR_DET_POSTPROCESSOR_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/postproc_base.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

class AnchorDetPostproc : public PostprocssBase {
public:
  explicit AnchorDetPostproc() = default;

  virtual bool process(const TensorData &, const AlgoPostprocParams &,
                       AlgoOutput &,
                       std::shared_ptr<RuntimeContext> &) const override;

  virtual bool batchProcess(const TensorData &, const AlgoPostprocParams &,
                            std::vector<AlgoOutput> &,
                            std::shared_ptr<RuntimeContext> &) const override;
};
} // namespace ai_core::dnn

#endif // __ANCHOR_DET_POSTPROCESSOR_HPP__
