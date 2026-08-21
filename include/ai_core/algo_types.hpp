#ifndef AI_CORE_ALGO_DATA_TYPES_HPP
#define AI_CORE_ALGO_DATA_TYPES_HPP

#include "ai_core/data_packet.hpp"
#include "ai_core/input_types.hpp"
#include "ai_core/output_types.hpp"
#include "ai_core/param_center.hpp"
#include "ai_core/postprocess_types.hpp"
#include "ai_core/preprocess_types.hpp"
#include <optional>

namespace ai_core {
/** Input variants accepted by bundled preprocessors. */
using AlgoInput =
    ParamCenter<std::variant<std::monostate, FrameInput, FrameInputWithMask>>;

/**
 * @brief Results produced by bundled postprocessors and an extension slot.
 *
 * An out-of-tree postprocessor can place its result in the trailing
 * `DataPacket` without changing this variant or misusing `RawModelOutput`.
 */
using AlgoOutput = ParamCenter<
    std::variant<std::monostate, ClsRet, DetRet, FprClsRet, RawModelOutput,
                 SegRet, DualRawSegRet, OCRRecoRet, DataPacket>>;

/** Parameter variants accepted by bundled preprocessors. */
using AlgoPreprocParams =
    ParamCenter<std::variant<std::monostate, FramePreprocessArg>>;

/** Parameter variants accepted by bundled postprocessors. */
using AlgoPostprocParams =
    ParamCenter<std::variant<std::monostate, AnchorDetParams, GenericPostParams,
                             ConfidenceFilterParams>>;

/** Type-erased construction parameters passed to plugin creators. */
using AlgoConstructParams = DataPacket;

/**
 * @brief Per-inference context flowing from preprocessor to postprocessor.
 *
 * The frame transform slots are typed: frame preprocessors fill them, and
 * postprocessors that map results back to source coordinates read them.
 * `extras` is the type-erased extension space for custom plugins. A context is
 * local to one synchronous inference call and does not outlive that call.
 */
struct RuntimeContext {
  std::optional<FrameTransformContext> frame_transform;
  std::vector<FrameTransformContext> frame_transform_batch;
  DataPacket extras;
};

/** Registry names selecting the three stages of a pipeline. */
struct AlgoModuleTypes {
  std::string preproc_module;
  std::string infer_module;
  std::string postproc_module;
};

} // namespace ai_core

#endif
