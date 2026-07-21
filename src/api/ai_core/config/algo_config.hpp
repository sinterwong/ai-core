/**
 * @file algo_config.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Optional configuration module (`ai_core::config`): load a full
 * algorithm pipeline definition from JSON, validated against a schema, so new
 * products can wire up algorithms without writing C++.
 * @version 1.0
 * @date 2026-07-18
 *
 * @copyright Copyright (c) 2026
 *
 */
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
 * @brief A fully parsed, ready-to-use pipeline definition. Feed the pieces
 * straight into AlgoInference / the standalone stages.
 */
struct AlgoConfig {
  std::string name;
  AlgoModuleTypes module_types;
  AlgoInferParams infer_params;

  // Optional stages: present only if the JSON supplied their params AND the
  // matching module name.
  bool has_preproc = false;
  bool has_postproc = false;
  AlgoPreprocParams preproc_params;
  AlgoPostprocParams postproc_params;
};

/**
 * @brief Parse + validate a config JSON file.
 *
 * @param config_path path to the .json file.
 * @param model_root  directory that `inferParams.modelPath` is resolved
 *   against. If empty, defaults to the config file's grandparent directory
 *   (so `<root>/conf/x.json` resolves `models/y.onnx` to `<root>/models/y.onnx`).
 * @throws ConfigError on any schema violation; std::runtime_error on I/O.
 */
AlgoConfig loadAlgoConfig(const std::string &config_path,
                          const std::string &model_root = "");

/**
 * @brief Parse + validate a config from an in-memory JSON string.
 * @param model_root resolves `modelPath` (no default derivation possible).
 */
AlgoConfig parseAlgoConfig(const std::string &json_text,
                           const std::string &model_root = "");

} // namespace ai_core::config

#endif // AI_CORE_CONFIG_ALGO_CONFIG_HPP
