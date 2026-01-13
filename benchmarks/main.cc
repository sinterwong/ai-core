#include "ai_core/logger.hpp"
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
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
