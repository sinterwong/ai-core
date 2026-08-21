#ifndef AI_CORE_CONFIG_ALGO_CONFIG_HPP
#define AI_CORE_CONFIG_ALGO_CONFIG_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/infer_config.hpp"

#include <stdexcept>
#include <string>

namespace ai_core::config {

/**
 * @brief Thrown on any schema violation, with a message naming the offending
 * key and what was expected vs found.
 */
class ConfigError : public std::runtime_error {
public:
  explicit ConfigError(const std::string &what) : std::runtime_error(what) {}
};

/**
 * @brief Validated pipeline definition ready for `AlgoInference`.
 */
struct AlgoConfig {
  std::string name;
  AlgoModuleTypes module_types;
  AlgoInferParams infer_params;

  // A stage is present only when both its module name and parameters were set.
  bool has_preproc = false;
  bool has_postproc = false;
  AlgoPreprocParams preproc_params;
  AlgoPostprocParams postproc_params;
};

/**
 * @brief Parse and validate a pipeline configuration file.
 *
 * @param config_path Path to the JSON file.
 * @param model_root Directory that `inferParams.modelPath` is resolved
 *   against. If empty, defaults to the config file's grandparent directory
 *   (so `<root>/conf/x.json` resolves `models/y.onnx` to
 * `<root>/models/y.onnx`).
 * @throws ConfigError on any schema violation; std::runtime_error on I/O.
 */
AlgoConfig loadAlgoConfig(const std::string &config_path,
                          const std::string &model_root = "");

/**
 * @brief Parse and validate an in-memory pipeline configuration.
 * @param model_root Base directory used to resolve `modelPath`; no path can be
 * inferred from an in-memory document.
 */
AlgoConfig parseAlgoConfig(const std::string &json_text,
                           const std::string &model_root = "");

} // namespace ai_core::config

#endif // AI_CORE_CONFIG_ALGO_CONFIG_HPP
