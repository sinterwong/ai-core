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

#include "ai_core/algo_infer_base.hpp" // Updated path
#include "ai_core/types/algo_data_types.hpp" // For AlgoConstructParams, AlgoInferParams, AlgoPostprocParams
#include "type_safe_factory.hpp"       // Assuming from common_utils
#include "vision_infer.hpp"            // Internal, assuming in src/
#include "vision_registrar.hpp"        // Internal, assuming in src/

#include "logger.hpp"                  // Assuming global from 3rdparty

namespace ai_core::dnn {
AlgoRegistrar::AlgoRegistrar() {
  vision::VisionRegistrar::getInstance();

  AlgoInferFactory::instance().registerCreator(
      "VisionInfer",
      [](const AlgoConstructParams &params) -> std::shared_ptr<AlgoInferBase> {
        // Make sure common_utils::ParamCenter/DataPacket placeholders are sufficient for getParam<T>
        auto moduleName = params.getParam<std::string>("moduleName");
        auto inferParam = params.getParam<AlgoInferParams>("inferParams");
        auto postproc = params.getParam<AlgoPostprocParams>("postProcParams");
        return std::make_shared<vision::VisionInfer>(moduleName, inferParam,
                                                     postproc);
      });
  LOG_INFOS << "Registered VisionInfer creator.";
}
} // namespace ai_core::dnn
