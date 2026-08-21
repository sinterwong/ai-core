#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace ai_core {

/** Severity ordered from most verbose to disabled. */
enum class LogLevel : std::uint8_t {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warning = 3,
  Error = 4,
  Fatal = 5,
  Off = 6
};

/** Process-wide settings for diagnostics emitted by ai-core and its plugins. */
struct LoggingConfig {
  LogLevel min_level = LogLevel::Debug;
  bool console_enabled = true;
  bool file_enabled = false;
  bool color_enabled = true;
  bool async_enabled = false;
  bool show_thread_id = true;
  bool show_source_location = true;
  bool show_category = true;
  bool json_output = false;

  std::string file_path = "app.log";
  std::size_t max_file_size = 10 * 1024 * 1024;
  int max_backup_count = 5;
  std::size_t async_queue_size = 8192;
  std::size_t flush_interval_ms = 1000;

  /** Tokens: %T time, %L level, %t thread, %s source, %c category, %m text. */
  std::string pattern = "[%T] [%L] [%t] [%s] %m";
};

/**
 * @brief A read-only view of one diagnostic passed to a user log handler.
 *
 * String views are valid only for the duration of the handler call. Copy any
 * field that must be retained.
 */
struct LogRecord {
  LogLevel level = LogLevel::Info;
  std::string_view message;
  std::chrono::system_clock::time_point timestamp;
  std::string_view file;
  std::string_view function;
  int line = 0;
  std::uint64_t thread_id = 0;
  std::string_view category;
};

using LogHandler = std::function<void(const LogRecord &)>;

/**
 * @brief Process-wide runtime controls for ai-core.
 *
 * The logging methods configure diagnostics produced internally by the core
 * library and maintained plugins. They do not expose the internal logger or
 * its logging macros. Configure logging during startup when possible.
 *
 * @par Thread safety
 * All methods may be called concurrently. A handler is invoked synchronously
 * on the emitting thread, or on the logging worker when async mode is enabled.
 */
class Runtime final {
public:
  Runtime() = delete;

  static void configureLogging(const LoggingConfig &config);
  [[nodiscard]] static LoggingConfig loggingConfig();

  /** Replace the downstream handler; pass an empty handler to remove it. */
  static void setLogHandler(LogHandler handler);

  /** Drain queued diagnostics and flush enabled sinks. */
  static void flushLogs();
};

} // namespace ai_core
