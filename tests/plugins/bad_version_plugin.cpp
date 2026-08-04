#include <ai_core/plugin_manager.hpp>

extern "C" AI_CORE_PLUGIN_EXPORT bool
ai_core_register_plugin_v1(ai_core::dnn::PluginRegistry &registry,
                           ai_core::dnn::PluginInfo &info) {
  info.api_version = ai_core::dnn::AI_CORE_PLUGIN_API_VERSION + 1;
  info.name = "test.bad-version";
  info.version = "0";
  return registry.registerInferenceEngine(
      "RollbackProbe", [](const ai_core::DataPacket &) {
        return std::shared_ptr<ai_core::dnn::IInferEnginePlugin>{};
      });
}
