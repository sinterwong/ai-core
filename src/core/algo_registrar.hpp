/**
 * @file vision_registrar.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __ALGO_REGISTRAR_HPP__
#define __ALGO_REGISTRAR_HPP__

#include "algo_infer_base.hpp"
#include "type_safe_factory.hpp"
namespace infer::dnn {

using AlgoInferFactory = common_utils::Factory<AlgoInferBase>;

class AlgoRegistrar {
public:
  static AlgoRegistrar &getInstance() {
    static AlgoRegistrar instance;
    return instance;
  }

  AlgoRegistrar(const AlgoRegistrar &) = delete;
  AlgoRegistrar &operator=(const AlgoRegistrar &) = delete;
  AlgoRegistrar(AlgoRegistrar &&) = delete;
  AlgoRegistrar &operator=(AlgoRegistrar &&) = delete;

private:
  AlgoRegistrar();
};

[[maybe_unused]] inline const static AlgoRegistrar &node_registrar =
    AlgoRegistrar::getInstance();

} // namespace infer::dnn

#endif