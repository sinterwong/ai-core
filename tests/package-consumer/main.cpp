#include <ai_core/runtime.hpp>
#include <ai_core/version.hpp>

static_assert(AI_CORE_VER_MAJOR == 2);

int main() {
  ai_core::LoggingConfig logging;
  logging.min_level = ai_core::LogLevel::Warning;
  logging.console_enabled = false;
  ai_core::Runtime::configureLogging(logging);
  ai_core::Runtime::setLogHandler({});
  ai_core::Runtime::flushLogs();
  return ai_core::Runtime::loggingConfig().console_enabled ? 1 : 0;
}
