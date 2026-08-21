#include "ai_core/plugin_manager.hpp"
#include "ai_core/runtime.hpp"
#include <benchmark/benchmark.h>

const static auto temp_init_log = []() {
  ai_core::LoggingConfig config;
  config.min_level = ai_core::LogLevel::Warning;
  config.console_enabled = true;
  config.file_enabled = false;
  ai_core::Runtime::configureLogging(config);
  return true;
}();

int main(int argc, char **argv) {
#ifdef AI_CORE_BENCH_PREPROC_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_BENCH_PREPROC_PLUGIN);
#endif
#ifdef AI_CORE_BENCH_POSTPROC_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_BENCH_POSTPROC_PLUGIN);
#endif
#ifdef AI_CORE_BENCH_ORT_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_BENCH_ORT_PLUGIN);
#endif
#ifdef AI_CORE_BENCH_NCNN_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_BENCH_NCNN_PLUGIN);
#endif
#ifdef AI_CORE_BENCH_TRT_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_BENCH_TRT_PLUGIN);
#endif
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
