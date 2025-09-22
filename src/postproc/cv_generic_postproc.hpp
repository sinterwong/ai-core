/**
 * @file cv_generic_postproc.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __CV_GENERIC_POSTPROC_HPP__
#define __CV_GENERIC_POSTPROC_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/postproc_base.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

class CVGenericPostproc : public IPostprocssPlugin {
public:
  explicit CVGenericPostproc() = default;

  virtual bool process(const TensorData &, const AlgoPostprocParams &,
                       AlgoOutput &,
                       std::shared_ptr<RuntimeContext> &) const override;

  virtual bool batchProcess(const TensorData &, const AlgoPostprocParams &,
                            std::vector<AlgoOutput> &,
                            std::shared_ptr<RuntimeContext> &) const override;
};
} // namespace ai_core::dnn

#endif // __I_IMAGE_PREPROCESSOR_HPP__
