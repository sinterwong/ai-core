/**
 * @file logger.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Lightweight logging interface. The implementation (file sinks, async
 * worker, formatting) lives entirely in logger.cpp so this header pulls in no
 * <iostream>/<fstream>/<thread>.
 * @version 0.2
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

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

// ============================================================================
// Log Level
// ============================================================================

enum class LogLevel : uint8_t {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warning = 3,
  Error = 4,
  Fatal = 5,
  Off = 6
};

// Compile-time log level threshold (define before including header to override)
#ifndef AI_CORE_LOG_LEVEL
#define AI_CORE_LOG_LEVEL 0 // Default: all levels enabled
#endif

constexpr LogLevel k_compile_time_log_level =
    static_cast<LogLevel>(AI_CORE_LOG_LEVEL);

[[nodiscard]] constexpr bool isLevelEnabled(LogLevel level) noexcept {
  return level >= k_compile_time_log_level;
}

// ============================================================================
// Source Location
// ============================================================================

struct SourceLocation {
  const char *file = "";
  const char *function = "";
  int line = 0;

  constexpr SourceLocation() noexcept = default;
  constexpr SourceLocation(const char *f, const char *fn, int l) noexcept
      : file(f), function(fn), line(l) {}
};

// ============================================================================
// Log Entry
// ============================================================================

struct LogEntry {
  LogLevel level = LogLevel::Info;
  std::string message;
  std::chrono::system_clock::time_point timestamp;
  SourceLocation location;
  uint64_t thread_id = 0;
  std::string_view category; // Non-owning reference, must outlive entry

  LogEntry() noexcept = default;

  // Defined in logger.cpp (captures timestamp + calling thread id).
  LogEntry(LogLevel lvl, std::string msg, SourceLocation loc,
           std::string_view cat = {}) noexcept;
};

// ============================================================================
// Logger Configuration
// ============================================================================

struct LoggerConfig {
  LogLevel min_level = LogLevel::Debug;
  bool console_enabled = true;
  bool file_enabled = false;
  bool color_enabled = true;
  bool async_enabled = false;
  bool show_thread_id = true;
  bool show_source_location = true;
  bool show_category = true;
  bool json_output = false; // Structured logging mode

  std::string file_path = "app.log";
  size_t max_file_size = 10 * 1024 * 1024; // 10MB
  int max_backup_count = 5;
  size_t async_queue_size = 8192;  // Ring buffer size for async mode
  size_t flush_interval_ms = 1000; // Periodic flush interval

  // Log pattern: %T=timestamp, %L=level, %t=thread, %s=source, %c=category,
  // %m=message
  std::string pattern = "[%T] [%L] [%t] [%s] %m";
};

// ============================================================================
// Logger Core (pimpl: all sinks and the async machinery live in logger.cpp)
// ============================================================================

class Logger {
public:
  using LogCallback = std::function<void(const LogEntry &)>;

  static Logger &instance() noexcept;

  // Non-copyable, non-movable
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger(Logger &&) = delete;
  Logger &operator=(Logger &&) = delete;

  // Configuration
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

  // Register custom log handler (e.g., for remote logging)
  void addCallback(LogCallback callback);
  void clearCallbacks() noexcept;

  // Core logging - use macros for source location capture
  void log(LogLevel level, std::string_view message, const SourceLocation &loc,
           std::string_view category = {});

  // Printf-style formatting
  template <typename... Args>
  void logf(LogLevel level, const SourceLocation &loc,
            std::string_view category, const char *fmt, Args &&...args);

  // Convenience methods
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

  // Flush all pending logs
  void flush();

  // Graceful shutdown
  void shutdown();

  // Check if level is enabled (hot path, inline & lock-free)
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

// ============================================================================
// Stream-Style Logger
// ============================================================================

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

  // Movable but not copyable
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

  // Manipulator support
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

// ============================================================================
// Printf-style Implementation
// ============================================================================

template <typename... Args>
void Logger::logf(LogLevel level, const SourceLocation &loc,
                  std::string_view category, const char *fmt, Args &&...args) {
  if (!isEnabled(level))
    return;

  // Calculate required size
  int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
  if (size <= 0)
    return;

  std::string buffer(static_cast<size_t>(size) + 1, '\0');
  std::snprintf(buffer.data(), buffer.size(), fmt, std::forward<Args>(args)...);
  buffer.resize(static_cast<size_t>(size));

  log(level, buffer, loc, category);
}

// ============================================================================
// Hex Dump Utility
// ============================================================================

std::string hexDump(const void *data, size_t size, size_t bytes_per_line = 16);

} // namespace ai_core::logging

// ============================================================================
// Logging Macros
// ============================================================================

// Internal macro for source location capture
#define AI_CORE_LOG_LOCATION                                                   \
  ::ai_core::logging::SourceLocation { __FILE__, __func__, __LINE__ }

// Check if log level is enabled at compile time
#define AI_CORE_LOG_ENABLED(level)                                             \
  (::ai_core::logging::isLevelEnabled(::ai_core::logging::LogLevel::level))

// Basic logging macros - message only
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

// Printf-style formatting macros
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

// Stream-style macros
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

// Category-based logging macros
#define LOG_CAT(level, cat, msg)                                               \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(level)) {                                \
      ::ai_core::logging::Logger::instance().log(                              \
          ::ai_core::logging::LogLevel::level, msg, AI_CORE_LOG_LOCATION,      \
          cat);                                                                \
    }                                                                          \
  } while (0)

// Conditional logging (only evaluates expression if level is enabled)
#define LOG_IF(level, condition, msg)                                          \
  do {                                                                         \
    if constexpr (AI_CORE_LOG_ENABLED(level)) {                                \
      if (condition) {                                                         \
        ::ai_core::logging::Logger::instance().log(                            \
            ::ai_core::logging::LogLevel::level, msg, AI_CORE_LOG_LOCATION);   \
      }                                                                        \
    }                                                                          \
  } while (0)

// Log once (useful for warnings that shouldn't repeat)
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

// Rate-limited logging (logs at most once per N milliseconds)
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
