/**
 * @file confidence_filter_postproc.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_CONFIDENCE_FILTER_POSTPROC_HPP
#define AI_CORE_CONFIDENCE_FILTER_POSTPROC_HPP

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"
#include "postproc_base.hpp"

namespace ai_core::dnn {

class ConfidenceFilterPostproc : public PostprocssBase {
public:
  explicit ConfidenceFilterPostproc() = default;

  virtual bool process(const TensorData &, const AlgoPostprocParams &,
                       AlgoOutput &,
                       std::shared_ptr<RuntimeContext> &) const override;

  virtual bool batchProcess(const TensorData &, const AlgoPostprocParams &,
                            std::vector<AlgoOutput> &,
                            std::shared_ptr<RuntimeContext> &) const override;
};
} // namespace ai_core::dnn

#endif // __I_IMAGE_PREPROCESSOR_HPP__
