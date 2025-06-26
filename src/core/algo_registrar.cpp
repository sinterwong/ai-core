/**
 * @file algo_registrar.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "algo_registrar.hpp"

#include "algo_infer_base.hpp"
#include "type_safe_factory.hpp"
#include "vision_infer.hpp"
#include "vision_registrar.hpp"

#include "logger.hpp"

namespace infer::dnn {
AlgoRegistrar::AlgoRegistrar() {
  vision::VisionRegistrar::getInstance();

  AlgoInferFactory::instance().registerCreator(
      "VisionInfer",
      [](const AlgoConstructParams &params) -> std::shared_ptr<AlgoInferBase> {
        auto moduleName = params.getParam<std::string>("moduleName");
        auto inferParam = params.getParam<AlgoInferParams>("inferParams");
        auto postproc = params.getParam<AlgoPostprocParams>("postProcParams");
        return std::make_shared<vision::VisionInfer>(moduleName, inferParam,
                                                     postproc);
      });
  LOG_INFOS << "Registered VisionInfer creator.";
}
} // namespace infer::dnn
