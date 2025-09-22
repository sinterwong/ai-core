#pragma once

#include "nlohmann/json.hpp"
#include <ai_core/algo_data_types.hpp>
#include <ai_core/infer_params_types.hpp>
#include <string>

namespace ai_core::example::utils {

struct AlgoConfigData {
  std::string moduleName;
  ai_core::AlgoModuleTypes modelTypes;
  ai_core::AlgoPreprocParams preprocParams;
  ai_core::AlgoInferParams inferParams;
  ai_core::AlgoPostprocParams postprocParams;
};

class AlgoConfigParser {
public:
  explicit AlgoConfigParser(const std::string &configPath);

  AlgoConfigData parse();

private:
  void loadAndValidateJson();

  ai_core::AlgoModuleTypes parseModuleTypes(const nlohmann::json &algoConfig);

  ai_core::AlgoPreprocParams
  parsePreprocParams(const nlohmann::json &algoConfig,
                     const std::string &preprocType);

  ai_core::AlgoInferParams parseInferParams(const nlohmann::json &algoConfig);

  ai_core::AlgoPostprocParams
  parsePostprocParams(const nlohmann::json &algoConfig,
                      const std::string &postprocType);

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