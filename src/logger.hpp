#pragma once

// Internal logging implementation shared by ai-core and maintained plugins.
// Downstream code configures it through <ai_core/runtime.hpp>.
#include "ai_core/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

namespace ai_core::logging {

using ::ai_core::LogLevel;
using LoggerConfig = ::ai_core::LoggingConfig;

// Define before including this header to remove lower levels at compile time.
#ifndef AI_CORE_LOG_LEVEL
#define AI_CORE_LOG_LEVEL 0
#endif

constexpr LogLevel k_compile_time_log_level =
    static_cast<LogLevel>(AI_CORE_LOG_LEVEL);

[[nodiscard]] constexpr bool isLevelEnabled(LogLevel level) noexcept {
  return level >= k_compile_time_log_level;
}

/** Source coordinates captured by the logging macros. */
struct SourceLocation {
  const char *file = "";
  const char *function = "";
  int line = 0;

  constexpr SourceLocation() noexcept = default;
  constexpr SourceLocation(const char *f, const char *fn, int l) noexcept
      : file(f), function(fn), line(l) {}
};

/** Fully attributed message delivered to logger sinks and callbacks. */
struct LogEntry {
  LogLevel level = LogLevel::Info;
  std::string message;
  std::chrono::system_clock::time_point timestamp;
  SourceLocation location;
  uint64_t thread_id = 0;
  std::string category;

  LogEntry() noexcept = default;

  /** Capture the current timestamp and calling thread identifier. */
  LogEntry(LogLevel lvl, std::string msg, SourceLocation loc,
           std::string_view cat = {}) noexcept;
};

/**
 * @brief Process-wide logger singleton.
 *
 * @par Thread safety
 * Logging and configuration methods may be called concurrently; sinks and the
 * asynchronous queue are synchronized internally. Configure sinks during
 * startup when possible so concurrent messages do not span configuration
 * changes.
 */
class Logger {
public:
  using LogCallback = std::function<void(const LogEntry &)>;

  static Logger &instance() noexcept;

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger(Logger &&) = delete;
  Logger &operator=(Logger &&) = delete;

  void configure(const LoggerConfig &config);
  [[nodiscard]] LoggerConfig config() const;

  void setLevel(LogLevel level) noexcept;
  [[nodiscard]] LogLevel level() const noexcept;

  void enableConsole(bool enable) noexcept;
  void enableFile(bool enable);
  void enableColor(bool enable) noexcept;
  void enableAsync(bool enable);
  void enableJson(bool enable) noexcept;

  void setFilePath(const std::string &path);
  void setPattern(const std::string &pattern);

  /** Replace the callback invoked for each accepted entry. */
  void setCallback(LogCallback callback);

  /** Emit one message; prefer the macros below when source capture is wanted.
   */
  void log(LogLevel level, std::string_view message, const SourceLocation &loc,
           std::string_view category = {});

  template <typename... Args>
  void logf(LogLevel level, const SourceLocation &loc,
            std::string_view category, const char *fmt, Args &&...args);

  void trace(std::string_view msg, const SourceLocation &loc = {},
             std::string_view cat = {});
  void debug(std::string_view msg, const SourceLocation &loc = {},
             std::string_view cat = {});
  void info(std::string_view msg, const SourceLocation &loc = {},
            std::string_view cat = {});
  void warning(std::string_view msg, const SourceLocation &loc = {},
               std::string_view cat = {});
  void error(std::string_view msg, const SourceLocation &loc = {},
             std::string_view cat = {});
  void fatal(std::string_view msg, const SourceLocation &loc = {},
             std::string_view cat = {});

  /** Drain the asynchronous queue and flush enabled sinks. */
  void flush();

  /** Stop the asynchronous worker after draining queued entries. */
  void shutdown();

  /** Lock-free runtime severity check for logging hot paths. */
  [[nodiscard]] bool isEnabled(LogLevel level) const noexcept {
    return level >= m_level.load(std::memory_order_relaxed);
  }

private:
  Logger();
  ~Logger();

  std::atomic<LogLevel> m_level{LogLevel::Debug};

  class Impl;
  std::unique_ptr<Impl> m_impl;
};

/** RAII stream that emits its accumulated message on destruction. */
class LogStream {
public:
  LogStream(Logger &logger, LogLevel level, SourceLocation loc,
            std::string_view category = {}) noexcept
      : m_logger(logger), m_level(level), m_location(loc), m_category(category),
        m_enabled(logger.isEnabled(level)) {}

  ~LogStream() {
    if (m_enabled) {
      m_logger.log(m_level, m_stream.str(), m_location, m_category);
    }
  }

  LogStream(LogStream &&) = default;
  LogStream &operator=(LogStream &&) = delete;
  LogStream(const LogStream &) = delete;
  LogStream &operator=(const LogStream &) = delete;

  template <typename T> LogStream &operator<<(const T &value) {
    if (m_enabled) {
      m_stream << value;
    }
    return *this;
  }

  LogStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
    if (m_enabled) {
      manip(m_stream);
    }
    return *this;
  }

private:
  Logger &m_logger;
  LogLevel m_level;
  SourceLocation m_location;
  std::string_view m_category;
  bool m_enabled;
  std::ostringstream m_stream;
};

template <typename... Args>
void Logger::logf(LogLevel level, const SourceLocation &loc,
                  std::string_view category, const char *fmt, Args &&...args) {
  if (!isEnabled(level))
    return;

  int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
  if (size <= 0)
    return;

  std::string buffer(static_cast<size_t>(size) + 1, '\0');
  std::snprintf(buffer.data(), buffer.size(), fmt, std::forward<Args>(args)...);
  buffer.resize(static_cast<size_t>(size));

  log(level, buffer, loc, category);
}

/** Render `size` bytes as offset-prefixed hexadecimal rows. */
std::string hexDump(const void *data, size_t size, size_t bytes_per_line = 16);

} // namespace ai_core::logging

// Source-location and compile-time filtering primitives.
#define AI_CORE_LOG_LOCATION                                                   \
  ::ai_core::logging::SourceLocation { __FILE__, __func__, __LINE__ }

#define AI_CORE_LOG_ENABLED(level)                                             \
  (::ai_core::logging::isLevelEnabled(::ai_core::logging::LogLevel::level))

// Message macros.
#define LOG_TRACE(msg)                                                         \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Trace)) {                                \
      ::ai_core::logging::Logger::instance().trace(msg, AI_CORE_LOG_LOCATION); \
    }                                                                          \
  } while (0)

#define LOG_DEBUG(msg)                                                         \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Debug)) {                                \
      ::ai_core::logging::Logger::instance().debug(msg, AI_CORE_LOG_LOCATION); \
    }                                                                          \
  } while (0)

#define LOG_INFO(msg)                                                          \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Info)) {                                 \
      ::ai_core::logging::Logger::instance().info(msg, AI_CORE_LOG_LOCATION);  \
    }                                                                          \
  } while (0)

#define LOG_WARNING(msg)                                                       \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Warning)) {                              \
      ::ai_core::logging::Logger::instance().warning(msg,                      \
                                                     AI_CORE_LOG_LOCATION);    \
    }                                                                          \
  } while (0)

#define LOG_ERROR(msg)                                                         \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Error)) {                                \
      ::ai_core::logging::Logger::instance().error(msg, AI_CORE_LOG_LOCATION); \
    }                                                                          \
  } while (0)

#define LOG_FATAL(msg)                                                         \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Fatal)) {                                \
      ::ai_core::logging::Logger::instance().fatal(msg, AI_CORE_LOG_LOCATION); \
    }                                                                          \
  } while (0)

// Printf-style macros.
#define LOG_TRACE_FMT(fmt, ...)                                                \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Trace)) {                                \
      ::ai_core::logging::Logger::instance().logf(                             \
          ::ai_core::logging::LogLevel::Trace, AI_CORE_LOG_LOCATION, {}, fmt,  \
          ##__VA_ARGS__);                                                      \
    }                                                                          \
  } while (0)

#define LOG_DEBUG_FMT(fmt, ...)                                                \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Debug)) {                                \
      ::ai_core::logging::Logger::instance().logf(                             \
          ::ai_core::logging::LogLevel::Debug, AI_CORE_LOG_LOCATION, {}, fmt,  \
          ##__VA_ARGS__);                                                      \
    }                                                                          \
  } while (0)

#define LOG_INFO_FMT(fmt, ...)                                                 \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Info)) {                                 \
      ::ai_core::logging::Logger::instance().logf(                             \
          ::ai_core::logging::LogLevel::Info, AI_CORE_LOG_LOCATION, {}, fmt,   \
          ##__VA_ARGS__);                                                      \
    }                                                                          \
  } while (0)

#define LOG_WARNING_FMT(fmt, ...)                                              \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Warning)) {                              \
      ::ai_core::logging::Logger::instance().logf(                             \
          ::ai_core::logging::LogLevel::Warning, AI_CORE_LOG_LOCATION, {},     \
          fmt, ##__VA_ARGS__);                                                 \
    }                                                                          \
  } while (0)

#define LOG_ERROR_FMT(fmt, ...)                                                \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Error)) {                                \
      ::ai_core::logging::Logger::instance().logf(                             \
          ::ai_core::logging::LogLevel::Error, AI_CORE_LOG_LOCATION, {}, fmt,  \
          ##__VA_ARGS__);                                                      \
    }                                                                          \
  } while (0)

#define LOG_FATAL_FMT(fmt, ...)                                                \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(Fatal)) {                                \
      ::ai_core::logging::Logger::instance().logf(                             \
          ::ai_core::logging::LogLevel::Fatal, AI_CORE_LOG_LOCATION, {}, fmt,  \
          ##__VA_ARGS__);                                                      \
    }                                                                          \
  } while (0)

// Stream-style macros.
#define LOG_TRACE_S                                                            \
  if constexpr (AI_CORE_LOG_ENABLED(Trace))                                    \
  ::ai_core::logging::LogStream(::ai_core::logging::Logger::instance(),        \
                                ::ai_core::logging::LogLevel::Trace,           \
                                AI_CORE_LOG_LOCATION)

#define LOG_DEBUG_S                                                            \
  if constexpr (AI_CORE_LOG_ENABLED(Debug))                                    \
  ::ai_core::logging::LogStream(::ai_core::logging::Logger::instance(),        \
                                ::ai_core::logging::LogLevel::Debug,           \
                                AI_CORE_LOG_LOCATION)

#define LOG_INFO_S                                                             \
  if constexpr (AI_CORE_LOG_ENABLED(Info))                                     \
  ::ai_core::logging::LogStream(::ai_core::logging::Logger::instance(),        \
                                ::ai_core::logging::LogLevel::Info,            \
                                AI_CORE_LOG_LOCATION)

#define LOG_WARNING_S                                                          \
  if constexpr (AI_CORE_LOG_ENABLED(Warning))                                  \
  ::ai_core::logging::LogStream(::ai_core::logging::Logger::instance(),        \
                                ::ai_core::logging::LogLevel::Warning,         \
                                AI_CORE_LOG_LOCATION)

#define LOG_ERROR_S                                                            \
  if constexpr (AI_CORE_LOG_ENABLED(Error))                                    \
  ::ai_core::logging::LogStream(::ai_core::logging::Logger::instance(),        \
                                ::ai_core::logging::LogLevel::Error,           \
                                AI_CORE_LOG_LOCATION)

#define LOG_FATAL_S                                                            \
  if constexpr (AI_CORE_LOG_ENABLED(Fatal))                                    \
  ::ai_core::logging::LogStream(::ai_core::logging::Logger::instance(),        \
                                ::ai_core::logging::LogLevel::Fatal,           \
                                AI_CORE_LOG_LOCATION)

// Category-aware logging.
#define LOG_CAT(level, cat, msg)                                               \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(level)) {                                \
      ::ai_core::logging::Logger::instance().log(                              \
          ::ai_core::logging::LogLevel::level, msg, AI_CORE_LOG_LOCATION,      \
          cat);                                                                \
    }                                                                          \
  } while (0)

// `condition` is evaluated only when the level is compiled in.
#define LOG_IF(level, condition, msg)                                          \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(level)) {                                \
      if (condition) {                                                         \
        ::ai_core::logging::Logger::instance().log(                            \
            ::ai_core::logging::LogLevel::level, msg, AI_CORE_LOG_LOCATION);   \
      }                                                                        \
    }                                                                          \
  } while (0)

// Emit once per call site.
#define LOG_ONCE(level, msg)                                                   \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(level)) {                                \
      static std::atomic<bool> logged{false};                                  \
      if (!logged.exchange(true, std::memory_order_relaxed)) {                 \
        ::ai_core::logging::Logger::instance().log(                            \
            ::ai_core::logging::LogLevel::level, msg, AI_CORE_LOG_LOCATION);   \
      }                                                                        \
    }                                                                          \
  } while (0)

// Emit at most once per call site within the requested interval.
#define LOG_EVERY_MS(level, ms, msg)                                           \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(level)) {                                \
      static std::atomic<int64_t> last_log_time{0};                            \
      auto now = std::chrono::steady_clock::now().time_since_epoch().count();  \
      auto last = last_log_time.load(std::memory_order_relaxed);               \
      if (now - last >= static_cast<int64_t>(ms) * 1000000) {                  \
        if (last_log_time.compare_exchange_strong(                             \
                last, now, std::memory_order_relaxed)) {                       \
          ::ai_core::logging::Logger::instance().log(                          \
              ::ai_core::logging::LogLevel::level, msg, AI_CORE_LOG_LOCATION); \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)
