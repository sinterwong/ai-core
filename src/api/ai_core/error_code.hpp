/**
 * @file error_code.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-01-13
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_INFER_ERROR_CODE_HPP
#define AI_CORE_INFER_ERROR_CODE_HPP
#include <cstdint>
#include <ostream>
#include <string_view>

namespace ai_core {

enum class InferErrorCode : int32_t {
  SUCCESS = 0,

  // init error
  InitFailed = 100,
  InitConfigFailed = 101,
  InitModelLoadFailed = 102,
  InitDeviceFailed = 103,
  InitMemoryAllocFailed = 104,
  InitDecryptionFailed = 105,
  NotInitialized = 106,
  InitRuntimeFailed = 107,
  InitEngineFailed = 108,
  InitContextFailed = 109,
  InitBindingFailed = 110,

  // infer error
  InferFailed = 200,
  InferInputError = 201,
  InferOutputError = 202,
  InferDeviceError = 203,
  InferPreprocessFailed = 204,
  InferMemoryError = 205,
  InferSetInputFailed = 206,
  InferExtractFailed = 207,
  InferUnsupportedOutputType = 208,
  InferTypeMismatch = 209,
  InferSizeMismatch = 210,
  InferInvalidInput = 211,
  InferExecutionFailed = 212,
  InferBindingError = 213,

  // async/stream error
  StreamCreationFailed = 250,
  StreamSyncFailed = 251,
  GraphCaptureFailed = 252,
  GraphLaunchFailed = 253,
  AsyncOperationPending = 254,

  // release error
  TerminateFailed = 300,

  // algo manager error
  AlgoNotFound = 400,
  AlgoRegisterFailed = 401,
  AlgoUnregisterFailed = 402,
  AlgoInferFailed = 403,
};

/**
 * @brief Stable, human-readable name for an error code. Never returns null;
 * unknown values yield "InferErrorCode(<n>)". Header-only (constexpr) so it
 * costs nothing and is available everywhere the enum is.
 */
constexpr std::string_view to_string(InferErrorCode code) noexcept {
  switch (code) {
  case InferErrorCode::SUCCESS:
    return "SUCCESS";
  case InferErrorCode::InitFailed:
    return "InitFailed";
  case InferErrorCode::InitConfigFailed:
    return "InitConfigFailed";
  case InferErrorCode::InitModelLoadFailed:
    return "InitModelLoadFailed";
  case InferErrorCode::InitDeviceFailed:
    return "InitDeviceFailed";
  case InferErrorCode::InitMemoryAllocFailed:
    return "InitMemoryAllocFailed";
  case InferErrorCode::InitDecryptionFailed:
    return "InitDecryptionFailed";
  case InferErrorCode::NotInitialized:
    return "NotInitialized";
  case InferErrorCode::InitRuntimeFailed:
    return "InitRuntimeFailed";
  case InferErrorCode::InitEngineFailed:
    return "InitEngineFailed";
  case InferErrorCode::InitContextFailed:
    return "InitContextFailed";
  case InferErrorCode::InitBindingFailed:
    return "InitBindingFailed";
  case InferErrorCode::InferFailed:
    return "InferFailed";
  case InferErrorCode::InferInputError:
    return "InferInputError";
  case InferErrorCode::InferOutputError:
    return "InferOutputError";
  case InferErrorCode::InferDeviceError:
    return "InferDeviceError";
  case InferErrorCode::InferPreprocessFailed:
    return "InferPreprocessFailed";
  case InferErrorCode::InferMemoryError:
    return "InferMemoryError";
  case InferErrorCode::InferSetInputFailed:
    return "InferSetInputFailed";
  case InferErrorCode::InferExtractFailed:
    return "InferExtractFailed";
  case InferErrorCode::InferUnsupportedOutputType:
    return "InferUnsupportedOutputType";
  case InferErrorCode::InferTypeMismatch:
    return "InferTypeMismatch";
  case InferErrorCode::InferSizeMismatch:
    return "InferSizeMismatch";
  case InferErrorCode::InferInvalidInput:
    return "InferInvalidInput";
  case InferErrorCode::InferExecutionFailed:
    return "InferExecutionFailed";
  case InferErrorCode::InferBindingError:
    return "InferBindingError";
  case InferErrorCode::StreamCreationFailed:
    return "StreamCreationFailed";
  case InferErrorCode::StreamSyncFailed:
    return "StreamSyncFailed";
  case InferErrorCode::GraphCaptureFailed:
    return "GraphCaptureFailed";
  case InferErrorCode::GraphLaunchFailed:
    return "GraphLaunchFailed";
  case InferErrorCode::AsyncOperationPending:
    return "AsyncOperationPending";
  case InferErrorCode::TerminateFailed:
    return "TerminateFailed";
  case InferErrorCode::AlgoNotFound:
    return "AlgoNotFound";
  case InferErrorCode::AlgoRegisterFailed:
    return "AlgoRegisterFailed";
  case InferErrorCode::AlgoUnregisterFailed:
    return "AlgoUnregisterFailed";
  case InferErrorCode::AlgoInferFailed:
    return "AlgoInferFailed";
  }
  return "InferErrorCode(unknown)";
}

inline std::ostream &operator<<(std::ostream &os, InferErrorCode code) {
  return os << to_string(code) << "(" << static_cast<int32_t>(code) << ")";
}

} // namespace ai_core
#endif
