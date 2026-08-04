#include "ai_core/logger.hpp"
#include "ai_core/plugin_manager.hpp"
#include <benchmark/benchmark.h>

// init log
const static auto temp_init_log = []() {
  ai_core::logging::Logger::instance().setLevel(
      ai_core::logging::LogLevel::Warning);
  ai_core::logging::Logger::instance().enableConsole(true);
  ai_core::logging::Logger::instance().enableFile(false);
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
