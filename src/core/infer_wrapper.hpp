/**
 * @file infer_wrapper.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-20
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __INFERENCE_SAFE_WRAPPER_HPP__
#define __INFERENCE_SAFE_WRAPPER_HPP__

#include "infer.hpp"
#include "infer_types.hpp"
#include <atomic>
#include <memory>

namespace infer {

template <typename AlgoClass, typename AlgoParams> class InferSafeWrapper {
public:
  InferSafeWrapper(const AlgoParams &param)
      : model(std::make_unique<AlgoClass>(param)), busy(false) {}

  InferErrorCode initialize() { return model->initialize(); }

  bool tryAcquire() {
    bool expected = false;
    return busy.compare_exchange_strong(expected, true);
  }

  void release() { busy.store(false); }

  dnn::Inference *getEngine() { return model.get(); }

private:
  std::unique_ptr<AlgoClass> model;
  std::atomic<bool> busy;
};

} // namespace infer

#endif