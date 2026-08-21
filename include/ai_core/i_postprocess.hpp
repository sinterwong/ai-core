#ifndef AI_CORE_POSTPROC_BASE_HPP
#define AI_CORE_POSTPROC_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

/**
 * @brief Postprocessing plugin interface.
 *
 * @par Thread safety
 * `process()` and `batchProcess()` must be reentrant. Implementations keep
 * per-call state in the supplied output and `RuntimeContext`; any shared cache
 * must be synchronized internally.
 */
class IPostprocessPlugin {
public:
  virtual ~IPostprocessPlugin() {};

  /** Decode one set of named model outputs into an algorithm result. */
  virtual InferErrorCode process(const TensorData &, const AlgoPostprocParams &,
                                 AlgoOutput &,
                                 std::shared_ptr<RuntimeContext> &) const = 0;

  /** Decode a batch into results ordered like the original inputs. */
  virtual InferErrorCode
  batchProcess(const TensorData &, const AlgoPostprocParams &,
               std::vector<AlgoOutput> &,
               std::shared_ptr<RuntimeContext> &) const = 0;
};
} // namespace ai_core::dnn

#endif
