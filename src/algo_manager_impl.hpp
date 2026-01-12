#ifndef AI_CORE_ALGO_MANAGER_IMPL_HPP
#define AI_CORE_ALGO_MANAGER_IMPL_HPP

#include "ai_core/algo_data_types.hpp"
#include "ai_core/algo_infer.hpp"
#include "ai_core/algo_manager.hpp"
#include "ai_core/infer_error_code.hpp"
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
                       AlgoPreprocParams &preproc_params, AlgoOutput &output,
                       AlgoPostprocParams &postproc_params);

  std::shared_ptr<AlgoInference> getAlgo(const std::string &name) const;

  bool hasAlgo(const std::string &name) const;

  void clear();

private:
  std::unordered_map<std::string, std::shared_ptr<AlgoInference>> m_algoMap;
  mutable std::shared_mutex m_mutex;
};

} // namespace ai_core::dnn

#endif
