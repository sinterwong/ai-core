#include <ai_core/plugin_manager.hpp>

#include <memory>

namespace {
class ExampleAmdInference final : public ai_core::dnn::IInferEnginePlugin {
public:
  explicit ExampleAmdInference(const ai_core::DataPacket &) {}

  ai_core::InferErrorCode initialize() override {
    return ai_core::InferErrorCode::SUCCESS;
  }
  ai_core::InferErrorCode infer(const ai_core::TensorData &inputs,
                                ai_core::TensorData &outputs) override {
    outputs = inputs;
    return ai_core::InferErrorCode::SUCCESS;
  }
  ai_core::InferErrorCode terminate() override {
    return ai_core::InferErrorCode::SUCCESS;
  }
  const ai_core::ModelInfo &getModelInfo() override { return info_; }

private:
  ai_core::ModelInfo info_;
};
} // namespace

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(ai_core::dnn::PluginRegistry &registry,
                           ai_core::dnn::PluginInfo &info) {
  info = {.name = "example.infer.amd",
          .version = "1.0.0",
          .provider = "example",
          .description = "Out-of-tree AMD inference plugin example",
          .capabilities = {"infer", "backend:amd"}};
  return registry.registerInferenceEngine(
      "ExampleAmdInference", [](const ai_core::DataPacket &params) {
        return std::make_shared<ExampleAmdInference>(params);
      });
}
