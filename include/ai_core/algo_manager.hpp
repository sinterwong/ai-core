#ifndef AI_CORE_ALGO_MANAGER_HPP
#define AI_CORE_ALGO_MANAGER_HPP

#include "ai_core/algo_inference.hpp"
#include "ai_core/error_code.hpp"
#include <memory>
#include <string>

#include "ai_core/algo_types.hpp"

namespace ai_core::dnn {

/**
 * @brief Registry of named AlgoInference instances.
 *
 * @par Thread safety
 * All methods may be called concurrently. Registry mutations take exclusive
 * access; lookups and dispatch share access. `infer()` also relies on the
 * selected `AlgoInference` instance's thread-safety contract.
 */
class AlgoManager : public std::enable_shared_from_this<AlgoManager> {
public:
  AlgoManager();
  ~AlgoManager();
  AlgoManager(const AlgoManager &) = delete;
  AlgoManager &operator=(const AlgoManager &) = delete;
  AlgoManager(AlgoManager &&) noexcept;
  AlgoManager &operator=(AlgoManager &&) noexcept;

  /** Register a non-null pipeline under a unique name. */
  InferErrorCode registerAlgo(const std::string &name,
                              const std::shared_ptr<AlgoInference> &algo);

  /** Remove `name`; removing an absent name is a successful no-op. */
  InferErrorCode unregisterAlgo(const std::string &name);

  /** Dispatch to `name`, or return `AlgoNotFound` when it is not registered. */
  InferErrorCode infer(const std::string &name, AlgoInput &input,
                       AlgoOutput &output);

  /** Return shared ownership of the named pipeline, or `nullptr` if absent. */
  std::shared_ptr<AlgoInference> getAlgo(const std::string &name) const;

  bool hasAlgo(const std::string &name) const;

  /** Remove all registry entries without terminating externally owned
   * pipelines. */
  void clear();

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};

} // namespace ai_core::dnn

#endif
