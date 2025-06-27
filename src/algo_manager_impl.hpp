#ifndef __AI_CORE_ALGO_MANAGER_IMPL_HPP_
#define __AI_CORE_ALGO_MANAGER_IMPL_HPP_

#include "ai_core/algo_infer.hpp"
#include "ai_core/algo_manager.hpp"
#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/infer_error_code.hpp"
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace ai_core::dnn {

class AlgoManager::Impl {
public:
  Impl();
  ~Impl();

  InferErrorCode registerAlgo(const std::string &name,
                              const std::shared_ptr<AlgoInference> &algo);

  InferErrorCode unregisterAlgo(const std::string &name);

  InferErrorCode infer(const std::string &name, AlgoInput &input,
                       AlgoPreprocParams &preprocParams, AlgoOutput &output,
                       AlgoPostprocParams &postprocParams);

  std::shared_ptr<AlgoInference> getAlgo(const std::string &name) const;

  bool hasAlgo(const std::string &name) const;

  void clear();

private:
  std::unordered_map<std::string, std::shared_ptr<AlgoInference>> algoMap_;
  mutable std::shared_mutex mutex_;
};

} // namespace ai_core::dnn

#endif // __AI_CORE_ALGO_MANAGER_IMPL_HPP_
