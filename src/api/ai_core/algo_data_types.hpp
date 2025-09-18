#ifndef __AI_CORE_ALGO_DATA_TYPES_HPP__
#define __AI_CORE_ALGO_DATA_TYPES_HPP__

#include "ai_core/algo_input_types.hpp"
#include "ai_core/algo_output_types.hpp"
#include "ai_core/data_packet.hpp"
#include "ai_core/param_center.hpp"
#include "ai_core/postproc_types.hpp"
#include "ai_core/preproc_types.hpp"

namespace ai_core {
// Algo input
using AlgoInput =
    ParamCenter<std::variant<std::monostate, FrameInput, BatchFrameInput,
                             FrameInputWithMask>>;

// Algo output
using AlgoOutput =
    ParamCenter<std::variant<std::monostate, ClsRet, DetRet, FprClsRet,
                             FeatureRet, SegRet, DaulRawSegRet, OCRRecoRet>>;

// Algo preproc params
using AlgoPreprocParams =
    ParamCenter<std::variant<std::monostate, FramePreprocessArg>>;

// Algo postproc params
using AlgoPostprocParams =
    ParamCenter<std::variant<std::monostate, AnchorDetParams, GenericPostParams,
                             ConfidenceFilterParams>>;

// Algo construct params
using AlgoConstructParams = DataPacket;

// Runtime Context
using RuntimeContext = DataPacket;

// Algo module types
struct AlgoModuleTypes {
  std::string preprocModule;
  std::string inferModule;
  std::string postprocModule;
};

} // namespace ai_core

#endif // __AI_CORE_ALGO_DATA_TYPES_HPP__
