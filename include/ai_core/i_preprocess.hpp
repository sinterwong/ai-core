#ifndef AI_CORE_PREPROC_BASE_HPP
#define AI_CORE_PREPROC_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

/**
 * @brief Preprocessing plugin interface.
 *
 * @par Thread safety
 * `process()` and `batchProcess()` must be reentrant. Implementations keep
 * per-call state in the supplied `TensorData` and `RuntimeContext`; any shared
 * cache must be synchronized internally.
 */
class IPreprocessPlugin {
public:
  virtual ~IPreprocessPlugin() {};

  /** Convert one algorithm input into the named tensors required by the model.
   */
  virtual InferErrorCode process(const AlgoInput &, const AlgoPreprocParams &,
                                 TensorData &,
                                 std::shared_ptr<RuntimeContext> &) const = 0;

  /** Convert a batch while preserving input order in the produced tensors. */
  virtual InferErrorCode
  batchProcess(const std::vector<AlgoInput> &, const AlgoPreprocParams &,
               TensorData &, std::shared_ptr<RuntimeContext> &) const = 0;
};
} // namespace ai_core::dnn

#endif
