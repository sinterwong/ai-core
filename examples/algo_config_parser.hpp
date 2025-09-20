#pragma once

#include "nlohmann/json.hpp"
#include <ai_core/algo_data_types.hpp>
#include <string>

namespace ai_core::example::utils {
class AlgoConfigParser {
public:
  explicit AlgoConfigParser(const std::string &configPath);

  ai_core::AlgoConstructParams parse();

private:
  void loadAndValidateJson();

  void parseCommonParams(const nlohmann::json &algoConfig,
                         ai_core::AlgoConstructParams &params);

  void parsePreprocParams(const nlohmann::json &algoConfig,
                          ai_core::AlgoConstructParams &params);

  void parseInferParams(const nlohmann::json &algoConfig,
                        ai_core::AlgoConstructParams &params);

  void parsePostprocParams(const nlohmann::json &algoConfig,
                           ai_core::AlgoConstructParams &params);

  ai_core::FramePreprocessArg
  parsePreprocFramePreprocessParams(const nlohmann::json &preprocJson);

  ai_core::GenericPostParams
  parsePostprocGenericParams(const nlohmann::json &postProcJson);

  ai_core::AnchorDetParams
  parsePostprocAnchorDetParams(const nlohmann::json &postProcJson);

  ai_core::ConfidenceFilterParams
  parsePostprocConfidenceFilterParams(const nlohmann::json &postProcJson);

  template <typename T>
  T getOptional(const nlohmann::json &j, const std::string &key,
                const T &defaultValue) {
    return j.contains(key) ? j.at(key).get<T>() : defaultValue;
  }

private:
  std::string mConfigPath;
  nlohmann::json mRootJson;
};

} // namespace ai_core::example::utils