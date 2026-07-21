/**
 * @file logger.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Full logger implementation: sinks, async worker, formatting. Nothing
 * here leaks into the public header.
 * @version 0.2
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "ai_core/logger.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ai_core::logging {

// ============================================================================
// Console Colors (ANSI escape codes)
// ============================================================================

namespace color {
inline constexpr std::string_view k_reset = "\033[0m";
inline constexpr std::string_view k_green = "\033[32m";
inline constexpr std::string_view k_yellow = "\033[33m";
inline constexpr std::string_view k_magenta = "\033[35m";
inline constexpr std::string_view k_cyan = "\033[36m";
inline constexpr std::string_view k_white = "\033[37m";
inline constexpr std::string_view k_bold_red = "\033[1;31m";
inline constexpr std::string_view k_gray = "\033[90m";
} // namespace color

// ============================================================================
// LogEntry
// ============================================================================

LogEntry::LogEntry(LogLevel lvl, std::string msg, SourceLocation loc,
                   std::string_view cat) noexcept
    : level(lvl), message(std::move(msg)),
      timestamp(std::chrono::system_clock::now()), location(loc),
      thread_id(std::hash<std::thread::id>{}(std::this_thread::get_id())),
      category(cat) {}

namespace {

// ============================================================================
// Lock-Free SPSC Ring Buffer for Async Logging
// ============================================================================

template <typename T> class SPSCRingBuffer {
public:
  explicit SPSCRingBuffer(size_t capacity)
      : m_capacity(nextPowerOf2(capacity)), m_mask(m_capacity - 1),
        m_buffer(std::make_unique<Slot[]>(m_capacity)) {}

  // Producer: try to push, returns false if full
  bool tryPush(T &&item) noexcept {
    const size_t head = m_head.load(std::memory_order_relaxed);
    const size_t next_head = (head + 1) & m_mask;

    if (next_head == m_tail.load(std::memory_order_acquire)) {
      return false; // Full
    }

    m_buffer[head].data = std::move(item);
    m_head.store(next_head, std::memory_order_release);
    return true;
  }

  // Consumer: try to pop, returns false if empty
  bool tryPop(T &item) noexcept {
    const size_t tail = m_tail.load(std::memory_order_relaxed);

    if (tail == m_head.load(std::memory_order_acquire)) {
      return false; // Empty
    }

    item = std::move(m_buffer[tail].data);
    m_tail.store((tail + 1) & m_mask, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool empty() const noexcept {
    return m_head.load(std::memory_order_acquire) ==
           m_tail.load(std::memory_order_acquire);
  }

private:
  static size_t nextPowerOf2(size_t n) noexcept {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
  }

  struct Slot {
    T data;
  };

  const size_t m_capacity;
  const size_t m_mask;
  std::unique_ptr<Slot[]> m_buffer;

  // Separate cache lines to avoid false sharing
  alignas(64) std::atomic<size_t> m_head{0};
  alignas(64) std::atomic<size_t> m_tail{0};
};

// ============================================================================
// Thread-Local Formatting Buffer
// ============================================================================

class FormatBuffer {
public:
  static constexpr size_t k_initial_capacity = 1024;

  FormatBuffer() { m_buffer.reserve(k_initial_capacity); }

  void clear() noexcept { m_buffer.clear(); }

  void append(std::string_view sv) { m_buffer.append(sv); }

  void append(char c) { m_buffer.push_back(c); }

  template <typename T> void appendNumber(T value) {
    // Fast integer to string conversion
    char temp[32];
    char *end = temp + sizeof(temp);
    char *ptr = end;

    if constexpr (std::is_signed_v<T>) {
      bool negative = value < 0;
      if (negative)
        value = -value;

      do {
        *--ptr = '0' + (value % 10);
        value /= 10;
      } while (value > 0);

      if (negative)
        *--ptr = '-';
    } else {
      do {
        *--ptr = '0' + (value % 10);
        value /= 10;
      } while (value > 0);
    }

    m_buffer.append(ptr, end - ptr);
  }

  void appendPadded(int value, int width, char pad = '0') {
    char temp[16];
    int len = 0;
    int v = value;

    do {
      temp[len++] = '0' + (v % 10);
      v /= 10;
    } while (v > 0);

    for (int i = len; i < width; ++i) {
      m_buffer.push_back(pad);
    }

    for (int i = len - 1; i >= 0; --i) {
      m_buffer.push_back(temp[i]);
    }
  }

  [[nodiscard]] std::string_view view() const noexcept {
    return {m_buffer.data(), m_buffer.size()};
  }

private:
  std::string m_buffer;
};

// Thread-local buffer accessor
FormatBuffer &getThreadLocalBuffer() {
  thread_local FormatBuffer buffer;
  buffer.clear();
  return buffer;
}

std::string_view levelName(LogLevel level) noexcept {
  switch (level) {
  case LogLevel::Trace:
    return "TRACE";
  case LogLevel::Debug:
    return "DEBUG";
  case LogLevel::Info:
    return "INFO";
  case LogLevel::Warning:
    return "WARNING";
  case LogLevel::Error:
    return "ERROR";
  case LogLevel::Fatal:
    return "FATAL";
  default:
    return "UNKNOWN";
  }
}

std::string_view levelColor(LogLevel level) noexcept {
  switch (level) {
  case LogLevel::Trace:
    return color::k_gray;
  case LogLevel::Debug:
    return color::k_cyan;
  case LogLevel::Info:
    return color::k_green;
  case LogLevel::Warning:
    return color::k_yellow;
  case LogLevel::Error:
    return color::k_bold_red;
  case LogLevel::Fatal:
    return color::k_magenta;
  default:
    return color::k_white;
  }
}

std::string_view extractFilename(std::string_view path) noexcept {
  auto pos = path.find_last_of("/\\");
  if (pos != std::string_view::npos) {
    return path.substr(pos + 1);
  }
  return path;
}

} // namespace

// ============================================================================
// Logger::Impl - all sinks and async machinery
// ============================================================================

class Logger::Impl {
public:
  // Configuration (atomic where possible for lock-free reads)
  std::atomic<bool> console_enabled{true};
  std::atomic<bool> file_enabled{false};
  std::atomic<bool> color_enabled{true};
  std::atomic<bool> async_enabled{false};
  std::atomic<bool> json_enabled{false};
  std::atomic<bool> show_thread_id{true};
  std::atomic<bool> show_source{true};
  std::atomic<bool> show_category{true};

  LoggerConfig config;
  mutable std::mutex config_mutex;

  // File output
  std::ofstream file_stream;
  std::mutex file_mutex;

  // Async logging
  std::unique_ptr<SPSCRingBuffer<LogEntry>> ring_buffer;
  std::unique_ptr<std::thread> worker_thread;
  std::atomic<bool> running{false};
  std::atomic<bool> initialized{false};
  std::condition_variable worker_cv;
  std::mutex worker_mutex;

  // Custom callbacks
  std::vector<Logger::LogCallback> callbacks;
  std::mutex callback_mutex;

  void processEntry(const LogEntry &entry) {
    if (console_enabled.load(std::memory_order_relaxed)) {
      writeConsole(entry);
    }

    if (file_enabled.load(std::memory_order_relaxed)) {
      writeFile(entry);
    }

    // Invoke callbacks
    {
      std::lock_guard lock(callback_mutex);
      for (const auto &cb : callbacks) {
        try {
          cb(entry);
        } catch (...) {
          // Ignore callback exceptions
        }
      }
    }
  }

  void writeConsole(const LogEntry &entry) {
    auto &buf = getThreadLocalBuffer();

    bool use_color = color_enabled.load(std::memory_order_relaxed);

    if (json_enabled.load(std::memory_order_relaxed)) {
      formatJson(buf, entry);
    } else {
      formatEntry(buf, entry, use_color);
    }

    // Error/Fatal to stderr, others to stdout
    if (entry.level >= LogLevel::Error) {
      std::cerr << buf.view() << '\n';
    } else {
      std::cout << buf.view() << '\n';
    }
  }

  void writeFile(const LogEntry &entry) {
    std::lock_guard lock(file_mutex);

    if (!file_stream.is_open())
      return;

    // Check rotation
    rotateFile();

    auto &buf = getThreadLocalBuffer();

    if (json_enabled.load(std::memory_order_relaxed)) {
      formatJson(buf, entry);
    } else {
      formatEntry(buf, entry, false); // No color for files
    }

    file_stream << buf.view() << '\n';

    // Flush for Error/Fatal levels immediately
    if (entry.level >= LogLevel::Error) {
      file_stream.flush();
    }
  }

  void asyncWorker() {
    LogEntry entry;
    std::vector<LogEntry> batch;
    batch.reserve(64);

    while (running.load(std::memory_order_acquire)) {
      // Wait for entries or shutdown signal
      {
        std::unique_lock lock(worker_mutex);
        worker_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
          return !ring_buffer->empty() ||
                 !running.load(std::memory_order_acquire);
        });
      }

      // Batch process entries for efficiency
      batch.clear();
      while (ring_buffer->tryPop(entry) && batch.size() < 64) {
        batch.push_back(std::move(entry));
      }

      for (const auto &e : batch) {
        processEntry(e);
      }
    }

    // Final drain on shutdown
    while (ring_buffer->tryPop(entry)) {
      processEntry(entry);
    }
  }

  void rotateFile() {
    if (!file_stream.is_open())
      return;

    size_t max_size;
    int max_backups;
    std::string path;
    {
      std::lock_guard lock(config_mutex);
      max_size = config.max_file_size;
      max_backups = config.max_backup_count;
      path = config.file_path;
    }

    auto pos = file_stream.tellp();
    if (pos < 0 || static_cast<size_t>(pos) < max_size) {
      return;
    }

    file_stream.close();

    namespace fs = std::filesystem;

    // Rotate: app.log.5 -> delete, app.log.4 -> app.log.5, ..., app.log ->
    // app.log.1
    for (int i = max_backups - 1; i >= 0; --i) {
      std::string old_name = path + (i > 0 ? "." + std::to_string(i) : "");
      std::string new_name = path + "." + std::to_string(i + 1);

      if (fs::exists(old_name)) {
        if (i == max_backups - 1) {
          fs::remove(old_name);
        } else {
          fs::rename(old_name, new_name);
        }
      }
    }

    file_stream.open(path, std::ios::out | std::ios::app);
  }

  void formatEntry(FormatBuffer &buf, const LogEntry &entry,
                   bool with_color) const {
    // Timestamp
    buf.append('[');
    formatTimestamp(buf, entry.timestamp);
    buf.append(']');

    // Level with optional color
    buf.append(" [");
    if (with_color) {
      buf.append(levelColor(entry.level));
    }

    auto level_str = levelName(entry.level);
    // Pad to 7 chars for alignment
    for (size_t i = level_str.size(); i < 7; ++i) {
      buf.append(' ');
    }
    buf.append(level_str);

    if (with_color) {
      buf.append(color::k_reset);
    }
    buf.append(']');

    // Thread ID
    if (show_thread_id.load(std::memory_order_relaxed)) {
      buf.append(" [T:");
      buf.appendNumber(entry.thread_id);
      buf.append(']');
    }

    // Category
    if (show_category.load(std::memory_order_relaxed) &&
        !entry.category.empty()) {
      buf.append(" [");
      buf.append(entry.category);
      buf.append(']');
    }

    // Source location
    if (show_source.load(std::memory_order_relaxed) &&
        entry.location.file != nullptr && entry.location.file[0] != '\0') {
      buf.append(" [");
      buf.append(extractFilename(entry.location.file));
      buf.append(':');
      buf.appendNumber(entry.location.line);

      if (entry.location.function != nullptr &&
          entry.location.function[0] != '\0') {
        buf.append(' ');
        buf.append(entry.location.function);
        buf.append("()");
      }
      buf.append(']');
    }

    // Message
    buf.append(' ');
    buf.append(entry.message);
  }

  void formatJson(FormatBuffer &buf, const LogEntry &entry) const {
    buf.append("{\"ts\":\"");
    formatTimestamp(buf, entry.timestamp);
    buf.append("\",\"level\":\"");
    buf.append(levelName(entry.level));
    buf.append("\"");

    if (show_thread_id.load(std::memory_order_relaxed)) {
      buf.append(",\"thread\":\"");
      buf.appendNumber(entry.thread_id);
      buf.append("\"");
    }

    if (show_category.load(std::memory_order_relaxed) &&
        !entry.category.empty()) {
      buf.append(",\"cat\":\"");
      buf.append(entry.category);
      buf.append("\"");
    }

    if (show_source.load(std::memory_order_relaxed) &&
        entry.location.file != nullptr && entry.location.file[0] != '\0') {
      buf.append(",\"file\":\"");
      buf.append(extractFilename(entry.location.file));
      buf.append("\",\"line\":");
      buf.appendNumber(entry.location.line);

      if (entry.location.function != nullptr &&
          entry.location.function[0] != '\0') {
        buf.append(",\"func\":\"");
        buf.append(entry.location.function);
        buf.append("\"");
      }
    }

    // Escape message for JSON
    buf.append(",\"msg\":\"");
    for (char c : entry.message) {
      switch (c) {
      case '"':
        buf.append("\\\"");
        break;
      case '\\':
        buf.append("\\\\");
        break;
      case '\n':
        buf.append("\\n");
        break;
      case '\r':
        buf.append("\\r");
        break;
      case '\t':
        buf.append("\\t");
        break;
      default:
        buf.append(c);
        break;
      }
    }
    buf.append("\"}");
  }

  void formatTimestamp(FormatBuffer &buf,
                       std::chrono::system_clock::time_point tp) const {
    auto time_t_val = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  tp.time_since_epoch()) %
              1000;

    std::tm tm_val;
#ifdef _WIN32
    localtime_s(&tm_val, &time_t_val);
#else
    localtime_r(&time_t_val, &tm_val);
#endif

    // YYYY-MM-DD HH:MM:SS.mmm
    buf.appendNumber(tm_val.tm_year + 1900);
    buf.append('-');
    buf.appendPadded(tm_val.tm_mon + 1, 2);
    buf.append('-');
    buf.appendPadded(tm_val.tm_mday, 2);
    buf.append(' ');
    buf.appendPadded(tm_val.tm_hour, 2);
    buf.append(':');
    buf.appendPadded(tm_val.tm_min, 2);
    buf.append(':');
    buf.appendPadded(tm_val.tm_sec, 2);
    buf.append('.');
    buf.appendPadded(static_cast<int>(ms.count()), 3);
  }
};

// ============================================================================
// Logger
// ============================================================================

Logger &Logger::instance() noexcept {
  static Logger logger;
  return logger;
}

Logger::Logger() : m_impl(std::make_unique<Impl>()) {
  m_impl->initialized.store(true, std::memory_order_release);

#ifdef _WIN32
  // Enable ANSI color support on Windows 10+
  HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode;
  if (GetConsoleMode(console, &mode)) {
    SetConsoleMode(console, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
  }
#endif
}

Logger::~Logger() { shutdown(); }

void Logger::configure(const LoggerConfig &config) {
  // Atomics can be set lock-free
  m_level.store(config.min_level, std::memory_order_relaxed);
  m_impl->console_enabled.store(config.console_enabled,
                                std::memory_order_relaxed);
  m_impl->color_enabled.store(config.color_enabled, std::memory_order_relaxed);
  m_impl->json_enabled.store(config.json_output, std::memory_order_relaxed);
  m_impl->show_thread_id.store(config.show_thread_id,
                               std::memory_order_relaxed);
  m_impl->show_source.store(config.show_source_location,
                            std::memory_order_relaxed);
  m_impl->show_category.store(config.show_category, std::memory_order_relaxed);

  bool need_file_reopen = false;
  bool need_async_change = false;

  {
    std::lock_guard lock(m_impl->config_mutex);
    need_file_reopen = (m_impl->config.file_path != config.file_path) ||
                       (m_impl->config.file_enabled != config.file_enabled);
    need_async_change = (m_impl->config.async_enabled != config.async_enabled);
    m_impl->config = config;
  }

  // Handle file changes
  if (need_file_reopen) {
    std::lock_guard lock(m_impl->file_mutex);

    if (m_impl->file_stream.is_open()) {
      m_impl->file_stream.flush();
      m_impl->file_stream.close();
    }

    if (config.file_enabled) {
      // Create parent directories if needed
      auto parent = std::filesystem::path(config.file_path).parent_path();
      if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
      }
      m_impl->file_stream.open(config.file_path, std::ios::out | std::ios::app);
    }

    m_impl->file_enabled.store(config.file_enabled, std::memory_order_relaxed);
  }

  // Handle async mode changes
  if (need_async_change) {
    if (config.async_enabled &&
        !m_impl->running.load(std::memory_order_acquire)) {
      // Start async worker
      m_impl->ring_buffer =
          std::make_unique<SPSCRingBuffer<LogEntry>>(config.async_queue_size);
      m_impl->running.store(true, std::memory_order_release);
      m_impl->worker_thread =
          std::make_unique<std::thread>([impl = m_impl.get()] {
            impl->asyncWorker();
          });
    } else if (!config.async_enabled &&
               m_impl->running.load(std::memory_order_acquire)) {
      // Stop async worker
      m_impl->running.store(false, std::memory_order_release);
      {
        std::lock_guard lock(m_impl->worker_mutex);
        m_impl->worker_cv.notify_one();
      }

      if (m_impl->worker_thread && m_impl->worker_thread->joinable()) {
        m_impl->worker_thread->join();
      }
      m_impl->worker_thread.reset();

      // Drain remaining entries
      if (m_impl->ring_buffer) {
        LogEntry entry;
        while (m_impl->ring_buffer->tryPop(entry)) {
          m_impl->processEntry(entry);
        }
        m_impl->ring_buffer.reset();
      }
    }

    m_impl->async_enabled.store(config.async_enabled,
                                std::memory_order_release);
  }
}

LoggerConfig Logger::config() const {
  std::lock_guard lock(m_impl->config_mutex);
  return m_impl->config;
}

void Logger::setLevel(LogLevel level) noexcept {
  m_level.store(level, std::memory_order_relaxed);
}

LogLevel Logger::level() const noexcept {
  return m_level.load(std::memory_order_relaxed);
}

void Logger::enableConsole(bool enable) noexcept {
  m_impl->console_enabled.store(enable, std::memory_order_relaxed);
}

void Logger::enableFile(bool enable) {
  if (enable == m_impl->file_enabled.load(std::memory_order_relaxed)) {
    return;
  }

  std::lock_guard lock(m_impl->file_mutex);

  if (enable && !m_impl->file_stream.is_open()) {
    std::string path;
    {
      std::lock_guard config_lock(m_impl->config_mutex);
      path = m_impl->config.file_path;
    }
    m_impl->file_stream.open(path, std::ios::out | std::ios::app);
  } else if (!enable && m_impl->file_stream.is_open()) {
    m_impl->file_stream.flush();
    m_impl->file_stream.close();
  }

  m_impl->file_enabled.store(enable, std::memory_order_relaxed);
}

void Logger::enableColor(bool enable) noexcept {
  m_impl->color_enabled.store(enable, std::memory_order_relaxed);
}

void Logger::enableAsync(bool enable) {
  LoggerConfig cfg = config();
  cfg.async_enabled = enable;
  configure(cfg);
}

void Logger::enableJson(bool enable) noexcept {
  m_impl->json_enabled.store(enable, std::memory_order_relaxed);
}

void Logger::setFilePath(const std::string &path) {
  std::lock_guard config_lock(m_impl->config_mutex);
  if (m_impl->config.file_path == path)
    return;

  m_impl->config.file_path = path;

  std::lock_guard file_lock(m_impl->file_mutex);
  if (m_impl->file_stream.is_open()) {
    m_impl->file_stream.flush();
    m_impl->file_stream.close();
  }

  if (m_impl->file_enabled.load(std::memory_order_relaxed)) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
      std::filesystem::create_directories(parent);
    }
    m_impl->file_stream.open(path, std::ios::out | std::ios::app);
  }
}

void Logger::setPattern(const std::string &pattern) {
  std::lock_guard lock(m_impl->config_mutex);
  m_impl->config.pattern = pattern;
}

void Logger::addCallback(LogCallback callback) {
  std::lock_guard lock(m_impl->callback_mutex);
  m_impl->callbacks.push_back(std::move(callback));
}

void Logger::clearCallbacks() noexcept {
  std::lock_guard lock(m_impl->callback_mutex);
  m_impl->callbacks.clear();
}

void Logger::log(LogLevel level, std::string_view message,
                 const SourceLocation &loc, std::string_view category) {
  if (!isEnabled(level))
    return;

  LogEntry entry(level, std::string(message), loc, category);

  if (m_impl->async_enabled.load(std::memory_order_acquire) &&
      m_impl->ring_buffer) {
    // Try lock-free push; fall back to sync if queue full
    if (!m_impl->ring_buffer->tryPush(std::move(entry))) {
      // Queue full - process synchronously to avoid log loss
      m_impl->processEntry(entry);
    } else {
      // Notify worker thread
      m_impl->worker_cv.notify_one();
    }
  } else {
    m_impl->processEntry(entry);
  }
}

void Logger::trace(std::string_view msg, const SourceLocation &loc,
                   std::string_view cat) {
  log(LogLevel::Trace, msg, loc, cat);
}

void Logger::debug(std::string_view msg, const SourceLocation &loc,
                   std::string_view cat) {
  log(LogLevel::Debug, msg, loc, cat);
}

void Logger::info(std::string_view msg, const SourceLocation &loc,
                  std::string_view cat) {
  log(LogLevel::Info, msg, loc, cat);
}

void Logger::warning(std::string_view msg, const SourceLocation &loc,
                     std::string_view cat) {
  log(LogLevel::Warning, msg, loc, cat);
}

void Logger::error(std::string_view msg, const SourceLocation &loc,
                   std::string_view cat) {
  log(LogLevel::Error, msg, loc, cat);
}

void Logger::fatal(std::string_view msg, const SourceLocation &loc,
                   std::string_view cat) {
  log(LogLevel::Fatal, msg, loc, cat);
}

void Logger::flush() {
  // Flush async queue first
  if (m_impl->async_enabled.load(std::memory_order_acquire) &&
      m_impl->ring_buffer) {
    // Signal worker and wait briefly for drain
    m_impl->worker_cv.notify_one();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Flush file
  {
    std::lock_guard lock(m_impl->file_mutex);
    if (m_impl->file_stream.is_open()) {
      m_impl->file_stream.flush();
    }
  }

  std::cout.flush();
  std::cerr.flush();
}

void Logger::shutdown() {
  if (!m_impl->initialized.exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  // Stop async worker
  if (m_impl->running.exchange(false, std::memory_order_acq_rel)) {
    m_impl->worker_cv.notify_all();

    if (m_impl->worker_thread && m_impl->worker_thread->joinable()) {
      m_impl->worker_thread->join();
    }
    m_impl->worker_thread.reset();

    // Drain remaining entries
    if (m_impl->ring_buffer) {
      LogEntry entry;
      while (m_impl->ring_buffer->tryPop(entry)) {
        m_impl->processEntry(entry);
      }
      m_impl->ring_buffer.reset();
    }
  }

  // Clear callbacks
  {
    std::lock_guard lock(m_impl->callback_mutex);
    m_impl->callbacks.clear();
  }

  // Close file
  {
    std::lock_guard lock(m_impl->file_mutex);
    if (m_impl->file_stream.is_open()) {
      m_impl->file_stream.flush();
      m_impl->file_stream.close();
    }
  }
}

// ============================================================================
// Hex Dump Utility
// ============================================================================

std::string hexDump(const void *data, size_t size, size_t bytes_per_line) {
  static constexpr char hex_chars[] = "0123456789ABCDEF";

  const auto *bytes = static_cast<const uint8_t *>(data);
  std::string result;
  result.reserve(size * 4); // Approximate

  for (size_t i = 0; i < size; i += bytes_per_line) {
    // Offset
    char offset_buf[32];
    std::snprintf(offset_buf, sizeof(offset_buf), "%08zx  ", i);
    result += offset_buf;

    // Hex bytes
    for (size_t j = 0; j < bytes_per_line; ++j) {
      if (i + j < size) {
        result += hex_chars[bytes[i + j] >> 4];
        result += hex_chars[bytes[i + j] & 0x0F];
        result += ' ';
      } else {
        result += "   ";
      }

      if (j == 7)
        result += ' '; // Extra space in middle
    }

    result += " |";

    // ASCII representation
    for (size_t j = 0; j < bytes_per_line && i + j < size; ++j) {
      char c = static_cast<char>(bytes[i + j]);
      result += (c >= 32 && c < 127) ? c : '.';
    }

    result += "|\n";
  }

  return result;
}

} // namespace ai_core::logging
