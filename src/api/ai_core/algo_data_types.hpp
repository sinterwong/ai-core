#ifndef __AI_CORE_ALGO_DATA_TYPES_HPP__
#define __AI_CORE_ALGO_DATA_TYPES_HPP__

#include "ai_core/algo_input_types.hpp"
#include "ai_core/algo_output_types.hpp"
#include "ai_core/postproc_types.hpp"
#include "ai_core/preproc_types.hpp"
#include "data_packet.hpp"
#include "param_center.hpp"

namespace ai_core {
// Algo input
using AlgoInput = common_utils::ParamCenter<
    std::variant<std::monostate, FrameInput, FrameInputWithMask>>;

// Algo output
using AlgoOutput = common_utils::ParamCenter<
    std::variant<std::monostate, ClsRet, DetRet, FprClsRet, FeatureRet,
                 DaulRawSegRet, BDiagSpecRet, TDiagSpecRet>>;

// Algo preproc params
using AlgoPreprocParams =
    common_utils::ParamCenter<std::variant<std::monostate, FramePreprocessArg>>;

// Algo postproc params
using AlgoPostprocParams = common_utils::ParamCenter<
    std::variant<std::monostate, AnchorDetParams, GenericPostParams>>;

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
