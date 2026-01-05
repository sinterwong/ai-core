#ifndef CUDA_UTILS_CUDA_DEVICE_BUFFER_HPP
#define CUDA_UTILS_CUDA_DEVICE_BUFFER_HPP

#include "cuda_helper.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace ai_core::cuda_utils {

template <typename T> class CudaHostBuffer;

/**
 * @brief 只读设备内存视图
 *
 * 用于将 buffer 作为 kernel 输入参数传递。
 * 轻量级，可按值传递。
 */
template <typename T> struct DeviceReadSpan {
  const T *ptr;
  size_t count;

  __host__ __device__ const T *data() const { return ptr; }
  __host__ __device__ size_t size() const { return count; }
  __host__ __device__ bool empty() const { return count == 0; }
  __host__ __device__ const T &operator[](size_t i) const { return ptr[i]; }
};

/**
 * @brief 可写设备内存视图
 *
 * 用于将 buffer 作为 kernel 输出参数传递。
 * 包含预期写入的元素数量，kernel 应写满这个数量。
 */
template <typename T> struct DeviceWriteSpan {
  T *ptr;
  size_t count;

  __host__ __device__ T *data() { return ptr; }
  __host__ __device__ const T *data() const { return ptr; }
  __host__ __device__ size_t size() const { return count; }
  __host__ __device__ bool empty() const { return count == 0; }
  __host__ __device__ T &operator[](size_t i) { return ptr[i]; }
  __host__ __device__ const T &operator[](size_t i) const { return ptr[i]; }
};

/**
 * @brief CUDA 设备内存管理类
 *
 * 设计理念：通过明确语义的 API 引导正确使用
 *
 * - 只读访问：使用 readSpan() 或 readPtr()
 * - 写入访问：使用 writeSpan(count) 或 writePtr(count)
 * - 追加写入：使用 appendSpan(count) 或 appendPtr(count)
 * - 不安全访问：使用 unsafePtr()（需手动调用 unsafeSetSize）
 */
template <typename T> class CudaDeviceBuffer {
public:
  CudaDeviceBuffer() : m_size(0), m_capacity(0), m_ptr(nullptr) {}

  explicit CudaDeviceBuffer(size_t size)
      : m_size(size), m_capacity(size), m_ptr(nullptr) {
    if (m_capacity > 0) {
      CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_capacity * sizeof(T)));
      CHECK_CUDA_ERROR(cudaMemset(m_ptr, 0, m_capacity * sizeof(T)));
    }
  }

  ~CudaDeviceBuffer() { freeMemory(); }

  // 禁止拷贝
  CudaDeviceBuffer(const CudaDeviceBuffer &) = delete;
  CudaDeviceBuffer &operator=(const CudaDeviceBuffer &) = delete;

  // 允许移动
  CudaDeviceBuffer(CudaDeviceBuffer &&other) noexcept
      : m_size(other.m_size), m_capacity(other.m_capacity), m_ptr(other.m_ptr) {
    other.m_size = 0;
    other.m_capacity = 0;
    other.m_ptr = nullptr;
  }

  CudaDeviceBuffer &operator=(CudaDeviceBuffer &&other) noexcept {
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
  /// 当前有效元素数量
  size_t size() const { return m_size; }

  /// 当前分配的容量
  size_t capacity() const { return m_capacity; }

  /// 有效数据的字节数
  size_t bytes() const { return m_size * sizeof(T); }

  /// 容量的字节数
  size_t capacityBytes() const { return m_capacity * sizeof(T); }

  /// 是否为空
  bool empty() const { return m_size == 0; }

  /// 是否已分配内存
  explicit operator bool() const { return m_ptr != nullptr; }

  // ==========================================================================
  // 只读访问 API（用于 Kernel 输入）
  // ==========================================================================
  /**
   * @brief 获取只读视图
   *
   * 用于将 buffer 作为 kernel 的输入参数。
   *
   * @code
   * auto input = buffer.readSpan();
   * myKernel<<<...>>>(input.data(), input.size(), ...);
   * @endcode
   */
  DeviceReadSpan<T> readSpan() const { return {m_ptr, m_size}; }

  /**
   * @brief 获取只读视图（指定范围）
   *
   * @param offset 起始偏移量（元素数）
   * @param count 元素数量
   */
  DeviceReadSpan<T> readSpan(size_t offset, size_t count) const {
    if (offset + count > m_size) {
      throw std::out_of_range("readSpan: range exceeds valid size");
    }
    return {m_ptr + offset, count};
  }

  /**
   * @brief 获取只读指针
   *
   * 便捷方法，等价于 readSpan().data()
   */
  const T *readPtr() const { return m_ptr; }

  // ==========================================================================
  // 写入访问 API（用于 Kernel 输出）
  // ==========================================================================
  /**
   * @brief 准备写入，获取可写视图
   *
   * 此方法会：
   * 1. 确保容量足够（必要时扩容）
   * 2. 立即将 size 设置为 count
   *
   * 调用者承诺会写满 count 个元素。
   *
   * @code
   * auto output = buffer.writeSpan(numElements);
   * myKernel<<<...>>>(input, output.data(), output.size());
   * // 无需手动设置 size
   * @endcode
   *
   * @param count 预期写入的元素数量
   * @param stream 用于可能的内存操作的 CUDA stream
   */
  DeviceWriteSpan<T> writeSpan(size_t count, cudaStream_t stream = 0) {
    prepareForWrite(count, stream);
    return {m_ptr, count};
  }

  /**
   * @brief 准备写入，获取可写指针
   *
   * 等价于 writeSpan(count).data()，但返回裸指针。
   * size 会立即更新为 count。
   *
   * @param count 预期写入的元素数量
   * @param stream 用于可能的内存操作的 CUDA stream
   */
  T *writePtr(size_t count, cudaStream_t stream = 0) {
    prepareForWrite(count, stream);
    return m_ptr;
  }

  /**
   * @brief 准备在指定偏移处写入
   *
   * 用于需要在特定位置写入的场景。
   * 会确保容量足够，并更新 size 为 offset + count。
   *
   * @param offset 写入的起始偏移量（元素数）
   * @param count 写入的元素数量
   * @param stream CUDA stream
   */
  DeviceWriteSpan<T> writeSpanAt(size_t offset, size_t count,
                                 cudaStream_t stream = 0) {
    size_t requiredSize = offset + count;
    if (requiredSize > m_capacity) {
      reallocate(requiredSize, true, stream);
    }
    m_size = std::max(m_size, requiredSize);
    return {m_ptr + offset, count};
  }

  // ==========================================================================
  // 追加写入 API
  // ==========================================================================
  /**
   * @brief 在现有数据末尾追加写入
   *
   * 返回从当前 size 位置开始的可写视图。
   * size 会更新为原 size + count。
   *
   * @code
   * // 第一批数据
   * buffer.writeSpan(100);
   * kernel1<<<...>>>(buffer.writePtr(100));
   *
   * // 追加更多数据
   * auto appendView = buffer.appendSpan(50);
   * kernel2<<<...>>>(appendView.data(), appendView.size());
   * // buffer.size() 现在是 150
   * @endcode
   *
   * @param count 追加的元素数量
   * @param stream CUDA stream
   */
  DeviceWriteSpan<T> appendSpan(size_t count, cudaStream_t stream = 0) {
    size_t offset = m_size;
    size_t newSize = m_size + count;

    if (newSize > m_capacity) {
      // 扩容策略：至少翻倍或满足需求
      size_t newCapacity = std::max(m_capacity * 2, newSize);
      reallocate(newCapacity, true, stream);
    }

    m_size = newSize;
    return {m_ptr + offset, count};
  }

  /**
   * @brief 追加写入，返回裸指针
   */
  T *appendPtr(size_t count, cudaStream_t stream = 0) {
    return appendSpan(count, stream).ptr;
  }

  // ==========================================================================
  // 不安全/低级访问（用于复杂场景）
  // ==========================================================================
  /**
   * @brief 获取裸指针（不安全）
   *
   * @warning 此方法不会自动更新 size。
   * 如果通过此指针写入了超过当前 size 的数据，
   * 必须调用 unsafeSetSize() 手动更新。
   *
   * 适用场景：
   * - 条件写入（实际写入量在运行时确定）
   * - 多次部分写入
   * - 与旧代码集成
   */
  T *unsafePtr() { return m_ptr; }
  const T *unsafePtr() const { return m_ptr; }

  /**
   * @brief 手动设置有效大小（不安全）
   *
   * 仅在使用 unsafePtr() 后需要更新 size 时使用。
   *
   * @param newSize 新的有效元素数量
   * @throw std::length_error 如果 newSize > capacity
   */
  void unsafeSetSize(size_t newSize) {
    if (newSize > m_capacity) {
      throw std::length_error(
          "unsafeSetSize: newSize exceeds capacity. Use reserve() first.");
    }
    m_size = newSize;
  }

  // ==========================================================================
  // 内存管理
  // ==========================================================================
  /**
   * @brief 预留容量
   *
   * 确保容量至少为 newCapacity，不改变 size。
   */
  void reserve(size_t newCapacity, cudaStream_t stream = 0) {
    if (newCapacity <= m_capacity) {
      return;
    }
    reallocate(newCapacity, true, stream);
  }

  /**
   * @brief 调整大小
   *
   * @param newSize 新的元素数量
   * @param preserveData 是否保留现有数据
   * @param stream CUDA stream
   */
  void resize(size_t newSize, bool preserveData = true,
              cudaStream_t stream = 0) {
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

    reallocate(newSize, preserveData, stream);
    m_size = newSize;
  }

  /**
   * @brief 清空数据（不释放内存）
   */
  void clear() { m_size = 0; }

  /**
   * @brief 释放多余内存，使 capacity == size
   */
  void shrinkToFit(cudaStream_t stream = 0) {
    if (m_capacity == m_size) {
      return;
    }

    if (m_size == 0) {
      freeMemory();
      m_capacity = 0;
      return;
    }

    T *newPtr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&newPtr, m_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(newPtr, m_ptr, m_size * sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream));

    freeMemory();
    m_ptr = newPtr;
    m_capacity = m_size;
  }

  /**
   * @brief 完全重置，释放所有内存
   */
  void reset() {
    freeMemory();
    m_size = 0;
    m_capacity = 0;
  }

  void swap(CudaDeviceBuffer &other) noexcept {
    std::swap(m_ptr, other.m_ptr);
    std::swap(m_size, other.m_size);
    std::swap(m_capacity, other.m_capacity);
  }

  // ==========================================================================
  // GPU 内存操作
  // ==========================================================================
  /**
   * @brief 将有效数据区域清零
   */
  void clearAsync(int byteValue = 0, cudaStream_t stream = 0) {
    if (m_size == 0 || !m_ptr)
      return;
    CHECK_CUDA_ERROR(cudaMemsetAsync(m_ptr, byteValue, bytes(), stream));
  }

  /**
   * @brief 清零指定范围
   */
  void clearRangeAsync(size_t offset, size_t count, int byteValue = 0,
                       cudaStream_t stream = 0) {
    if (count == 0)
      return;
    if (offset + count > m_size) {
      throw std::out_of_range("clearRangeAsync: range exceeds valid size");
    }
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(m_ptr + offset, byteValue, count * sizeof(T), stream));
  }

  // ==========================================================================
  // Host <-> Device 数据传输
  // ==========================================================================
  // -------------------- Host -> Device (初始化/覆盖) --------------------

  /**
   * @brief 从 host 数据初始化 buffer
   *
   * 完全覆盖 buffer 内容，size 设置为 count。
   */
  void initFromHost(const T *srcPtr, size_t count, cudaStream_t stream = 0) {
    if (count == 0) {
      m_size = 0;
      return;
    }

    if (count > m_capacity) {
      reallocate(count, false, stream);
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_ptr, srcPtr, count * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    m_size = count;
  }

  void initFromHost(const std::vector<T> &src, cudaStream_t stream = 0) {
    initFromHost(src.data(), src.size(), stream);
  }

  void initFromHost(const CudaHostBuffer<T> &src, cudaStream_t stream = 0) {
    initFromHost(src.readPtr(), src.size(), stream);
  }

  // -------------------- Host -> Device (写入到指定位置) --------------------

  /**
   * @brief 从 host 写入到指定位置
   *
   * @param srcPtr 源数据指针
   * @param dstOffset 目标偏移量（元素数）
   * @param count 元素数量
   * @param stream CUDA stream
   */
  void writeFromHost(const T *srcPtr, size_t dstOffset, size_t count,
                     cudaStream_t stream = 0) {
    if (count == 0)
      return;

    size_t requiredSize = dstOffset + count;
    if (requiredSize > m_capacity) {
      throw std::out_of_range(
          "writeFromHost: exceeds capacity. Call reserve() first.");
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_ptr + dstOffset, srcPtr,
                                     count * sizeof(T), cudaMemcpyHostToDevice,
                                     stream));
    m_size = std::max(m_size, requiredSize);
  }

  // -------------------- Device -> Host --------------------

  /**
   * @brief 读取所有有效数据到 host
   */
  void readToHost(T *dstPtr, cudaStream_t stream = 0) const {
    if (m_size == 0)
      return;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dstPtr, m_ptr, m_size * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream));
  }

  /**
   * @brief 读取指定范围到 host
   */
  void readToHost(T *dstPtr, size_t srcOffset, size_t count,
                  cudaStream_t stream = 0) const {
    if (count == 0)
      return;
    if (srcOffset + count > m_size) {
      throw std::out_of_range("readToHost: range exceeds valid size");
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dstPtr, m_ptr + srcOffset,
                                     count * sizeof(T), cudaMemcpyDeviceToHost,
                                     stream));
  }

  /**
   * @brief 读取到 vector（同步操作）
   */
  std::vector<T> toVector(cudaStream_t stream = 0) const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      readToHost(result.data(), stream);
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    return result;
  }

  /**
   * @brief 读取到 CudaHostBuffer
   *
   * 会调整目标 buffer 的 size 以匹配源数据。
   */
  void readToHost(CudaHostBuffer<T> &dst, cudaStream_t stream = 0) const {
    if (m_size == 0) {
      dst.clear();
      return;
    }

    if (m_size > dst.capacity()) {
      dst.reserve(m_size);
    }

    readToHost(dst.writePtr(m_size), stream);
  }

  // -------------------- Device -> Device --------------------

  /**
   * @brief 从另一个 buffer 初始化
   */
  void initFromDevice(const CudaDeviceBuffer<T> &src, cudaStream_t stream = 0) {
    if (src.empty()) {
      m_size = 0;
      return;
    }

    if (src.size() > m_capacity) {
      reallocate(src.size(), false, stream);
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_ptr, src.readPtr(), src.bytes(),
                                     cudaMemcpyDeviceToDevice, stream));
    m_size = src.size();
  }

  /**
   * @brief 从另一个 buffer 的指定范围拷贝
   */
  void writeFromDevice(const CudaDeviceBuffer<T> &src, size_t srcOffset,
                       size_t dstOffset, size_t count,
                       cudaStream_t stream = 0) {
    if (count == 0)
      return;

    if (srcOffset + count > src.size()) {
      throw std::out_of_range("writeFromDevice: source range out of bounds");
    }

    size_t requiredSize = dstOffset + count;
    if (requiredSize > m_capacity) {
      throw std::out_of_range(
          "writeFromDevice: exceeds capacity. Call reserve() first.");
    }

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(m_ptr + dstOffset, src.readPtr() + srcOffset,
                        count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    m_size = std::max(m_size, requiredSize);
  }

private:
  void prepareForWrite(size_t count, cudaStream_t stream) {
    if (count > m_capacity) {
      reallocate(count, false, stream); // 写入操作不需要保留旧数据
    }
    m_size = count;
  }

  void reallocate(size_t newCapacity, bool preserveData, cudaStream_t stream) {
    T *newPtr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&newPtr, newCapacity * sizeof(T)));

    if (preserveData && m_ptr && m_size > 0) {
      size_t copyCount = std::min(m_size, newCapacity);
      CHECK_CUDA_ERROR(cudaMemcpyAsync(newPtr, m_ptr, copyCount * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream));
    }

    freeMemory();
    m_ptr = newPtr;
    m_capacity = newCapacity;
  }

  void freeMemory() {
    if (m_ptr) {
      cudaFree(m_ptr);
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
void swap(CudaDeviceBuffer<T> &a, CudaDeviceBuffer<T> &b) noexcept {
  a.swap(b);
}

using DeviceByteBuffer = CudaDeviceBuffer<uint8_t>;

} // namespace ai_core::cuda_utils

#endif // CUDA_UTILS_CUDA_DEVICE_BUFFER_HPP