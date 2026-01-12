#ifndef AI_CORE_INFER_ERROR_CODE_HPP
#define AI_CORE_INFER_ERROR_CODE_HPP
#include <cstdint>

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

} // namespace ai_core
#endif
