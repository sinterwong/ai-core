#ifndef CUDA_UTILS_CUDA_HOST_BUFFER_HPP
#define CUDA_UTILS_CUDA_HOST_BUFFER_HPP

#include "cuda_helper.cuh"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace ai_core::cuda_utils {

// 前置声明
template <typename T> class CudaDeviceBuffer;

/**
 * @brief 只读 Host 内存视图
 *
 * 用于传递 host buffer 的只读引用，支持迭代器。
 */
template <typename T> struct HostReadSpan {
  const T *ptr;
  size_t count;

  const T *data() const { return ptr; }
  size_t size() const { return count; }
  bool empty() const { return count == 0; }
  const T &operator[](size_t i) const { return ptr[i]; }

  const T *begin() const { return ptr; }
  const T *end() const { return ptr + count; }
  const T *cbegin() const { return ptr; }
  const T *cend() const { return ptr + count; }
};

/**
 * @brief 可写 Host 内存视图
 *
 * 用于传递 host buffer 的可写引用。
 */
template <typename T> struct HostWriteSpan {
  T *ptr;
  size_t count;

  T *data() { return ptr; }
  const T *data() const { return ptr; }
  size_t size() const { return count; }
  bool empty() const { return count == 0; }
  T &operator[](size_t i) { return ptr[i]; }
  const T &operator[](size_t i) const { return ptr[i]; }

  T *begin() { return ptr; }
  T *end() { return ptr + count; }
  const T *begin() const { return ptr; }
  const T *end() const { return ptr + count; }
};

/**
 * @brief CUDA 锁页内存（Pinned Memory）管理类
 *
 * 设计理念：与 CudaDeviceBuffer 保持一致的语义化 API
 *
 * 锁页内存特点：
 * - Host 端可直接读写（支持下标访问）
 * - 与 GPU 之间的 DMA 传输更快
 * - 支持异步传输（cudaMemcpyAsync）
 * - 分配/释放开销比普通内存大
 *
 * API 分类：
 * - 只读访问：readSpan() / readPtr()
 * - 写入访问：writeSpan(count) / writePtr(count)
 * - 追加写入：appendSpan(count) / appendPtr(count)
 * - 直接元素访问：operator[]（Host 端特有）
 * - 不安全访问：unsafePtr() + unsafeSetSize()
 */
template <typename T> class CudaHostBuffer {
public:
  // ==========================================================================
  // 构造与析构
  // ==========================================================================

  CudaHostBuffer() : m_size(0), m_capacity(0), m_ptr(nullptr) {}

  explicit CudaHostBuffer(size_t size)
      : m_size(size), m_capacity(size), m_ptr(nullptr) {
    if (m_capacity > 0) {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_capacity * sizeof(T)));
      std::memset(m_ptr, 0, m_capacity * sizeof(T));
    }
  }

  /**
   * @brief 从初始化列表构造
   */
  CudaHostBuffer(std::initializer_list<T> init)
      : m_size(init.size()), m_capacity(init.size()), m_ptr(nullptr) {
    if (m_capacity > 0) {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_capacity * sizeof(T)));
      std::copy(init.begin(), init.end(), m_ptr);
    }
  }

  /**
   * @brief 从迭代器范围构造
   */
  template <typename InputIt>
  CudaHostBuffer(InputIt first, InputIt last)
      : m_size(0), m_capacity(0), m_ptr(nullptr) {
    size_t count = std::distance(first, last);
    if (count > 0) {
      m_size = count;
      m_capacity = count;
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_capacity * sizeof(T)));
      std::copy(first, last, m_ptr);
    }
  }

  ~CudaHostBuffer() { freeMemory(); }

  // 禁止拷贝
  CudaHostBuffer(const CudaHostBuffer &) = delete;
  CudaHostBuffer &operator=(const CudaHostBuffer &) = delete;

  // 允许移动
  CudaHostBuffer(CudaHostBuffer &&other) noexcept
      : m_size(other.m_size), m_capacity(other.m_capacity), m_ptr(other.m_ptr) {
    other.m_size = 0;
    other.m_capacity = 0;
    other.m_ptr = nullptr;
  }

  CudaHostBuffer &operator=(CudaHostBuffer &&other) noexcept {
    if (this != &other) {
      freeMemory();
      m_size = other.m_size;
      m_capacity = other.m_capacity;
      m_ptr = other.m_ptr;

      other.m_size = 0;
      other.m_capacity = 0;
      other.m_ptr = nullptr;
    }
    return *this;
  }

  // ==========================================================================
  // 属性查询
  // ==========================================================================

  size_t size() const { return m_size; }
  size_t capacity() const { return m_capacity; }
  size_t bytes() const { return m_size * sizeof(T); }
  size_t capacityBytes() const { return m_capacity * sizeof(T); }
  bool empty() const { return m_size == 0; }
  explicit operator bool() const { return m_ptr != nullptr; }

  // ==========================================================================
  // 只读访问 API
  // ==========================================================================

  /**
   * @brief 获取只读视图
   */
  HostReadSpan<T> readSpan() const { return {m_ptr, m_size}; }

  /**
   * @brief 获取指定范围的只读视图
   */
  HostReadSpan<T> readSpan(size_t offset, size_t count) const {
    if (offset + count > m_size) {
      throw std::out_of_range("readSpan: range exceeds valid size");
    }
    return {m_ptr + offset, count};
  }

  /**
   * @brief 获取只读指针
   */
  const T *readPtr() const { return m_ptr; }

  // ==========================================================================
  // 写入访问 API
  // ==========================================================================

  /**
   * @brief 准备写入，获取可写视图
   *
   * 会确保容量足够，并立即将 size 设置为 count。
   *
   * @param count 预期写入的元素数量
   */
  HostWriteSpan<T> writeSpan(size_t count) {
    prepareForWrite(count);
    return {m_ptr, count};
  }

  /**
   * @brief 准备写入，获取可写指针
   */
  T *writePtr(size_t count) {
    prepareForWrite(count);
    return m_ptr;
  }

  /**
   * @brief 在指定偏移处准备写入
   */
  HostWriteSpan<T> writeSpanAt(size_t offset, size_t count) {
    size_t requiredSize = offset + count;
    if (requiredSize > m_capacity) {
      reallocate(requiredSize, true);
    }
    m_size = std::max(m_size, requiredSize);
    return {m_ptr + offset, count};
  }

  // ==========================================================================
  // 追加写入 API
  // ==========================================================================

  /**
   * @brief 在末尾追加空间
   *
   * @param count 追加的元素数量
   * @return 指向追加区域的可写视图
   */
  HostWriteSpan<T> appendSpan(size_t count) {
    size_t offset = m_size;
    size_t newSize = m_size + count;

    if (newSize > m_capacity) {
      size_t newCapacity = std::max(m_capacity * 2, newSize);
      reallocate(newCapacity, true);
    }

    m_size = newSize;
    return {m_ptr + offset, count};
  }

  T *appendPtr(size_t count) { return appendSpan(count).ptr; }

  // ==========================================================================
  // 直接元素访问（Host Buffer 特有）
  // ==========================================================================

  /**
   * @brief 下标访问（带边界检查）
   */
  T &at(size_t index) {
    if (index >= m_size) {
      throw std::out_of_range("CudaHostBuffer::at: index out of range");
    }
    return m_ptr[index];
  }

  const T &at(size_t index) const {
    if (index >= m_size) {
      throw std::out_of_range("CudaHostBuffer::at: index out of range");
    }
    return m_ptr[index];
  }

  /**
   * @brief 下标访问（不检查边界）
   */
  T &operator[](size_t index) { return m_ptr[index]; }
  const T &operator[](size_t index) const { return m_ptr[index]; }

  /**
   * @brief 首元素引用
   */
  T &front() { return m_ptr[0]; }
  const T &front() const { return m_ptr[0]; }

  /**
   * @brief 尾元素引用
   */
  T &back() { return m_ptr[m_size - 1]; }
  const T &back() const { return m_ptr[m_size - 1]; }

  // ==========================================================================
  // STL 风格迭代器
  // ==========================================================================

  T *begin() { return m_ptr; }
  T *end() { return m_ptr + m_size; }
  const T *begin() const { return m_ptr; }
  const T *end() const { return m_ptr + m_size; }
  const T *cbegin() const { return m_ptr; }
  const T *cend() const { return m_ptr + m_size; }

  // ==========================================================================
  // 不安全/低级访问
  // ==========================================================================

  /**
   * @brief 获取裸指针（不安全）
   *
   * @warning 通过此指针的写入不会自动更新 size
   */
  T *unsafePtr() { return m_ptr; }
  const T *unsafePtr() const { return m_ptr; }

  /**
   * @brief 手动设置有效大小（不安全）
   */
  void unsafeSetSize(size_t newSize) {
    if (newSize > m_capacity) {
      throw std::length_error("unsafeSetSize: newSize exceeds capacity");
    }
    m_size = newSize;
  }

  // ==========================================================================
  // 内存管理
  // ==========================================================================

  void reserve(size_t newCapacity) {
    if (newCapacity <= m_capacity) {
      return;
    }
    reallocate(newCapacity, true);
  }

  void resize(size_t newSize, bool preserveData = true) {
    if (newSize == m_size) {
      return;
    }

    if (newSize == 0) {
      m_size = 0;
      return;
    }

    if (newSize <= m_capacity) {
      m_size = newSize;
      return;
    }

    reallocate(newSize, preserveData);
    m_size = newSize;
  }

  /**
   * @brief 调整大小并用指定值填充新增区域
   */
  void resize(size_t newSize, const T &value) {
    size_t oldSize = m_size;
    resize(newSize, true);

    // 填充新增部分
    if (newSize > oldSize) {
      std::fill(m_ptr + oldSize, m_ptr + newSize, value);
    }
  }

  void clear() { m_size = 0; }

  void shrinkToFit() {
    if (m_capacity == m_size) {
      return;
    }

    if (m_size == 0) {
      freeMemory();
      m_capacity = 0;
      return;
    }

    T *newPtr = nullptr;
    CHECK_CUDA_ERROR(cudaMallocHost(&newPtr, m_size * sizeof(T)));
    std::memcpy(newPtr, m_ptr, m_size * sizeof(T));

    freeMemory();
    m_ptr = newPtr;
    m_capacity = m_size;
  }

  void reset() {
    freeMemory();
    m_size = 0;
    m_capacity = 0;
  }

  void swap(CudaHostBuffer &other) noexcept {
    std::swap(m_ptr, other.m_ptr);
    std::swap(m_size, other.m_size);
    std::swap(m_capacity, other.m_capacity);
  }

  // ==========================================================================
  // STL 风格修改操作
  // ==========================================================================

  /**
   * @brief 在末尾添加元素
   */
  void push_back(const T &value) {
    if (m_size >= m_capacity) {
      size_t newCapacity = m_capacity == 0 ? 16 : m_capacity * 2;
      reallocate(newCapacity, true);
    }
    m_ptr[m_size++] = value;
  }

  void push_back(T &&value) {
    if (m_size >= m_capacity) {
      size_t newCapacity = m_capacity == 0 ? 16 : m_capacity * 2;
      reallocate(newCapacity, true);
    }
    m_ptr[m_size++] = std::move(value);
  }

  /**
   * @brief 移除末尾元素
   */
  void pop_back() {
    if (m_size > 0) {
      --m_size;
    }
  }

  /**
   * @brief 原地构造元素
   */
  template <typename... Args> T &emplace_back(Args &&...args) {
    if (m_size >= m_capacity) {
      size_t newCapacity = m_capacity == 0 ? 16 : m_capacity * 2;
      reallocate(newCapacity, true);
    }
    new (&m_ptr[m_size]) T(std::forward<Args>(args)...);
    return m_ptr[m_size++];
  }

  // ==========================================================================
  // Host 端内存操作
  // ==========================================================================

  /**
   * @brief 用指定值填充有效区域
   */
  void fill(const T &value) { std::fill(m_ptr, m_ptr + m_size, value); }

  /**
   * @brief 将有效区域清零
   */
  void zero() {
    if (m_size > 0) {
      std::memset(m_ptr, 0, m_size * sizeof(T));
    }
  }

  // ==========================================================================
  // 与其他容器的数据交换
  // ==========================================================================

  /**
   * @brief 从 vector 初始化
   */
  void initFromVector(const std::vector<T> &src) {
    if (src.empty()) {
      m_size = 0;
      return;
    }

    if (src.size() > m_capacity) {
      reallocate(src.size(), false);
    }

    std::memcpy(m_ptr, src.data(), src.size() * sizeof(T));
    m_size = src.size();
  }

  /**
   * @brief 转换为 vector
   */
  std::vector<T> toVector() const {
    return std::vector<T>(m_ptr, m_ptr + m_size);
  }

  /**
   * @brief 赋值操作符（从 vector）
   */
  CudaHostBuffer &operator=(const std::vector<T> &src) {
    initFromVector(src);
    return *this;
  }

  // ==========================================================================
  // Host <-> Device 数据传输
  // ==========================================================================

  /**
   * @brief 从 Device buffer 读取数据
   *
   * 会调整本 buffer 的 size 以匹配源数据。
   */
  void readFromDevice(const CudaDeviceBuffer<T> &src, cudaStream_t stream = 0) {
    if (src.empty()) {
      m_size = 0;
      return;
    }

    if (src.size() > m_capacity) {
      reallocate(src.size(), false);
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_ptr, src.readPtr(), src.bytes(),
                                     cudaMemcpyDeviceToHost, stream));
    m_size = src.size();
  }

  /**
   * @brief 从 Device buffer 读取指定范围
   */
  void readFromDevice(const CudaDeviceBuffer<T> &src, size_t srcOffset,
                      size_t dstOffset, size_t count, cudaStream_t stream = 0) {
    if (count == 0)
      return;

    if (srcOffset + count > src.size()) {
      throw std::out_of_range("readFromDevice: source range out of bounds");
    }

    size_t requiredSize = dstOffset + count;
    if (requiredSize > m_capacity) {
      throw std::out_of_range(
          "readFromDevice: exceeds capacity. Call reserve() first.");
    }

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(m_ptr + dstOffset, src.readPtr() + srcOffset,
                        count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    m_size = std::max(m_size, requiredSize);
  }

  /**
   * @brief 写入到 Device buffer
   *
   * 便捷方法，等价于 dst.initFromHost(*this)
   */
  void writeToDevice(CudaDeviceBuffer<T> &dst, cudaStream_t stream = 0) const {
    dst.initFromHost(m_ptr, m_size, stream);
  }

  /**
   * @brief 异步写入到 Device buffer（指定范围）
   */
  void writeToDeviceAsync(CudaDeviceBuffer<T> &dst, size_t srcOffset,
                          size_t dstOffset, size_t count,
                          cudaStream_t stream = 0) const {
    if (count == 0)
      return;

    if (srcOffset + count > m_size) {
      throw std::out_of_range("writeToDeviceAsync: source range out of bounds");
    }

    dst.writeFromHost(m_ptr + srcOffset, dstOffset, count, stream);
  }

  /**
   * @brief 将数据拷贝到外部 vector，自动处理 resize
   */
  void copyTo(std::vector<T> &dest) const {
    if (dest.size() != m_size) {
      dest.resize(m_size); // 只有在大小改变时才会重新分配内存
    }
    if (m_size > 0) {
      std::memcpy(dest.data(), m_ptr, m_size * sizeof(T));
    }
  }

private:
  void prepareForWrite(size_t count) {
    if (count > m_capacity) {
      reallocate(count, false);
    }
    m_size = count;
  }

  void reallocate(size_t newCapacity, bool preserveData) {
    T *newPtr = nullptr;
    CHECK_CUDA_ERROR(cudaMallocHost(&newPtr, newCapacity * sizeof(T)));

    if (preserveData && m_ptr && m_size > 0) {
      size_t copyCount = std::min(m_size, newCapacity);
      std::memcpy(newPtr, m_ptr, copyCount * sizeof(T));
    }

    freeMemory();
    m_ptr = newPtr;
    m_capacity = newCapacity;
  }

  void freeMemory() {
    if (m_ptr) {
      cudaFreeHost(m_ptr);
      m_ptr = nullptr;
    }
  }

private:
  size_t m_size;
  size_t m_capacity;
  T *m_ptr;
};

// ADL swap
template <typename T>
void swap(CudaHostBuffer<T> &a, CudaHostBuffer<T> &b) noexcept {
  a.swap(b);
}

} // namespace ai_core::cuda_utils

#endif // CUDA_UTILS_CUDA_HOST_BUFFER_HPP