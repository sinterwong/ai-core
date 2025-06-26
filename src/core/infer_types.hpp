/**
 * @file types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-11-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __INFERENCE_TYPES_HPP__
#define __INFERENCE_TYPES_HPP__

#include <string>

#include "algo_input_types.hpp"
#include "algo_output_types.hpp"
#include "data_packet.hpp"
#include "param_center.hpp"
#include "infer_params_types.hpp"
#include "postprocess_types.hpp"
#include "typed_buffer.hpp"

namespace infer {

enum class InferErrorCode : int32_t {
  SUCCESS = 0,

  // init error
  INIT_FAILED = 100,
  INIT_CONFIG_FAILED = 101,
  INIT_MODEL_LOAD_FAILED = 102,
  INIT_DEVICE_FAILED = 103,
  INIT_MEMORY_ALLOC_FAILED = 104,
  INIT_DECRYPTION_FAILED = 105,
  NOT_INITIALIZED = 106,

  // infer error
  INFER_FAILED = 200,
  INFER_INPUT_ERROR = 201,
  INFER_OUTPUT_ERROR = 202,
  INFER_DEVICE_ERROR = 203,
  INFER_PREPROCESS_FAILED = 204,
  INFER_MEMORY_ERROR = 205,
  INFER_SET_INPUT_FAILED = 206,
  INFER_EXTRACT_FAILED = 207,
  INFER_UNSUPPORTED_OUTPUT_TYPE = 208,

  // release error
  TERMINATE_FAILED = 300,

  // algo manager error
  ALGO_NOT_FOUND = 400,
  ALGO_REGISTER_FAILED = 401,
  ALGO_UNREGISTER_FAILED = 402,
  ALGO_INFER_FAILED = 403,
};

// Algo input
using AlgoInput = common_utils::ParamCenter<std::variant<std::monostate, FrameInput>>;

// Model output(after infering, before postprocess)
struct ModelOutput {
  std::map<std::string, TypedBuffer> outputs;
  std::map<std::string, std::vector<int>> outputShapes;
};

// Algo output
using AlgoOutput = common_utils::ParamCenter<
    std::variant<std::monostate, ClsRet, DetRet, FprClsRet, FeatureRet>>;

// Algo postproc params
using AlgoPostprocParams =
    common_utils::ParamCenter<std::variant<std::monostate, AnchorDetParams>>;

// Algo infer params
using AlgoInferParams =
    common_utils::ParamCenter<std::variant<std::monostate, FrameInferParam>>;

using AlgoConstructParams = common_utils::DataPacket;

} // namespace infer
#endif
