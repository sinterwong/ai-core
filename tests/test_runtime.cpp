#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/runtime.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace ai_core::test {
namespace {

TEST(RuntimeLoggingTest, ConfiguresAndForwardsInternalDiagnostics) {
  const LoggingConfig original = Runtime::loggingConfig();

  LoggingConfig config;
  config.min_level = LogLevel::Trace;
  config.console_enabled = false;
  config.file_enabled = false;
  config.color_enabled = false;
  config.show_source_location = false;
  config.pattern = "[%L] %m";
  Runtime::configureLogging(config);

  const LoggingConfig active = Runtime::loggingConfig();
  EXPECT_EQ(active.min_level, LogLevel::Trace);
  EXPECT_FALSE(active.console_enabled);
  EXPECT_FALSE(active.file_enabled);
  EXPECT_FALSE(active.color_enabled);
  EXPECT_FALSE(active.show_source_location);
  EXPECT_EQ(active.pattern, "[%L] %m");

  std::string diagnostic;
  Runtime::setLogHandler([&diagnostic](const LogRecord &record) {
    diagnostic.assign(record.message);
  });

  dnn::AlgoPreproc preprocessor("not_initialized");
  AlgoInput input;
  TensorData output;
  std::shared_ptr<RuntimeContext> context = std::make_shared<RuntimeContext>();
  EXPECT_EQ(preprocessor.process(input, output, context),
            InferErrorCode::NotInitialized);
  EXPECT_NE(diagnostic.find("Preprocessor is not initialized"),
            std::string::npos);

  Runtime::setLogHandler({});
  Runtime::configureLogging(original);
}

} // namespace
} // namespace ai_core::test
