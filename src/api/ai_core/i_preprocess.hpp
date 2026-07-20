/**
 * @file i_preprocess.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_PREPROC_BASE_HPP
#define AI_CORE_PREPROC_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

/**
 * @brief Preprocessing plugin interface.
 *
 * @par Thread-safety contract
 * process / batchProcess are const and must be reentrant: implementations keep
 * no mutable per-call state on the object (all scratch flows through the
 * passed-in TensorData / RuntimeContext), so one instance serves concurrent
 * calls. Any internal cache must be synchronized by the implementation.
 */
class IPreprocessPlugin {
public:
  virtual ~IPreprocessPlugin() {};

  virtual InferErrorCode process(const AlgoInput &, const AlgoPreprocParams &,
                                 TensorData &,
                                 std::shared_ptr<RuntimeContext> &) const = 0;

  virtual InferErrorCode
  batchProcess(const std::vector<AlgoInput> &, const AlgoPreprocParams &,
               TensorData &, std::shared_ptr<RuntimeContext> &) const = 0;
};
} // namespace ai_core::dnn

#endif