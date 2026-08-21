
#ifndef AI_CORE_INFER_PARAMS_TYPES_HPP
#define AI_CORE_INFER_PARAMS_TYPES_HPP

#include "ai_core/common_types.hpp"
#include <map>
#include <string>

namespace ai_core {
/** Backend-independent model and execution configuration. */
struct AlgoInferParams {
  std::string name;           ///< Logical model name used in diagnostics.
  std::string model_path;     ///< Model file path interpreted by the backend.
  bool need_decrypt = false;  ///< Decrypt the model before backend loading.
  std::string decryptkey_str; ///< Key used only when `need_decrypt` is true.
  DeviceType device_type;
  DataType data_type;
  /** Per-output capacity in bytes for backends that require caller sizing. */
  std::map<std::string, size_t> max_output_buffer_sizes;

  /** ORT intra-op thread count; zero leaves the choice to ONNX Runtime. */
  int intra_op_num_threads = 0;
  /** ORT inter-op thread count; zero leaves the choice to ONNX Runtime. */
  int inter_op_num_threads = 0;
};
} // namespace ai_core

#endif
