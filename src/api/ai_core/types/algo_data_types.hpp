#ifndef __AI_CORE_ALGO_DATA_TYPES_HPP__
#define __AI_CORE_ALGO_DATA_TYPES_HPP__

#include "ai_core/types/algo_input_types.hpp"
#include "ai_core/types/algo_output_types.hpp"
#include "ai_core/types/postprocess_types.hpp"
#include "data_packet.hpp"
#include "param_center.hpp"

namespace ai_core {
// Algo input
using AlgoInput =
    common_utils::ParamCenter<std::variant<std::monostate, FrameInput>>;

// Algo output
using AlgoOutput = common_utils::ParamCenter<
    std::variant<std::monostate, ai_core::ClsRet, ai_core::DetRet,
                 ai_core::FprClsRet, ai_core::FeatureRet>>;

// Algo preproc params
using AlgoPreprocParams =
    common_utils::ParamCenter<std::variant<std::monostate, FramePreprocessArg>>;

// Algo postproc params
using AlgoPostprocParams =
    common_utils::ParamCenter<std::variant<std::monostate, AnchorDetParams>>;

// Algo construct params
using AlgoConstructParams = common_utils::DataPacket;

// Algo module types
struct AlgoModuleTypes {
  std::string preprocModule;
  std::string inferModule;
  std::string postprocModule;
};

} // namespace ai_core

#endif // __AI_CORE_ALGO_DATA_TYPES_HPP__
