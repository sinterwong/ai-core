#ifndef AI_CORE_ALGO_DATA_TYPES_HPP
#define AI_CORE_ALGO_DATA_TYPES_HPP

#include "ai_core/algo_input_types.hpp"
#include "ai_core/algo_output_types.hpp"
#include "ai_core/data_packet.hpp"
#include "ai_core/param_center.hpp"
#include "ai_core/postproc_types.hpp"
#include "ai_core/preproc_types.hpp"

namespace ai_core {
// Algo input
using AlgoInput =
    ParamCenter<std::variant<std::monostate, FrameInput, FrameInputWithMask>>;

// Algo output
using AlgoOutput = ParamCenter<
    std::variant<std::monostate, ClsRet, DetRet, FprClsRet, RawModelOutput,
                 SegRet, DaulRawSegRet, OCRRecoRet>>;

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
  std::string preproc_module;
  std::string infer_module;
  std::string postproc_module;
};

} // namespace ai_core

#endif
