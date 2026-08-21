#ifndef CUDA_UTILS_CUDA_DEVICE_BUFFER_HPP
#define CUDA_UTILS_CUDA_DEVICE_BUFFER_HPP

#include "cuda_helper.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace ai_core::cuda_utils {

template <typename T> class CudaHostBuffer;

/** Non-owning device-memory range passed to read-only kernel parameters. */
template <typename T> struct DeviceReadSpan {
  const T *ptr;
  size_t count;

  __host__ __device__ const T *data() const { return ptr; }
  __host__ __device__ size_t size() const { return count; }
  __host__ __device__ bool empty() const { return count == 0; }
  __host__ __device__ const T &operator[](size_t i) const { return ptr[i]; }
};

/** Non-owning device-memory range that a kernel is expected to fill. */
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
 * @brief Move-only CUDA device allocation with vector-like size and capacity.
 *
 * `size()` is the initialized or reserved-for-write prefix; `capacity()` is
 * the allocated element count. Methods accepting a CUDA stream may enqueue
 * work, so referenced host/device memory must remain valid until that stream
 * completes.
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

  CudaDeviceBuffer(const CudaDeviceBuffer &) = delete;
  CudaDeviceBuffer &operator=(const CudaDeviceBuffer &) = delete;

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

  size_t size() const { return m_size; }

  size_t capacity() const { return m_capacity; }

  size_t bytes() const { return m_size * sizeof(T); }

  size_t capacityBytes() const { return m_capacity * sizeof(T); }

  bool empty() const { return m_size == 0; }

  explicit operator bool() const { return m_ptr != nullptr; }

  DeviceReadSpan<T> readSpan() const { return {m_ptr, m_size}; }

  DeviceReadSpan<T> readSpan(size_t offset, size_t count) const {
    if (offset + count > m_size) {
      throw std::out_of_range("readSpan: range exceeds valid size");
    }
    return {m_ptr + offset, count};
  }

  const T *readPtr() const { return m_ptr; }

  /**
   * @brief Reserve `count` output elements and mark them as the valid range.
   *
   * Existing data may be discarded if growth reallocates the buffer.
   */
  DeviceWriteSpan<T> writeSpan(size_t count, cudaStream_t stream = 0) {
    prepareForWrite(count, stream);
    return {m_ptr, count};
  }

  T *writePtr(size_t count, cudaStream_t stream = 0) {
    prepareForWrite(count, stream);
    return m_ptr;
  }

  DeviceWriteSpan<T> writeSpanAt(size_t offset, size_t count,
                                 cudaStream_t stream = 0) {
    size_t required_size = offset + count;
    if (required_size > m_capacity) {
      reallocate(required_size, true, stream);
    }
    m_size = std::max(m_size, required_size);
    return {m_ptr + offset, count};
  }

  DeviceWriteSpan<T> appendSpan(size_t count, cudaStream_t stream = 0) {
    size_t offset = m_size;
    size_t new_size = m_size + count;

    if (new_size > m_capacity) {
      size_t new_capacity = std::max(m_capacity * 2, new_size);
      reallocate(new_capacity, true, stream);
    }

    m_size = new_size;
    return {m_ptr + offset, count};
  }

  T *appendPtr(size_t count, cudaStream_t stream = 0) {
    return appendSpan(count, stream).ptr;
  }

  /**
   * @brief Return the allocation without changing its valid size.
   *
   * A caller writing beyond `size()` must subsequently call
   * `unsafeSetSize()` before exposing the data.
   */
  T *unsafePtr() { return m_ptr; }
  const T *unsafePtr() const { return m_ptr; }

  void unsafeSetSize(size_t new_size) {
    if (new_size > m_capacity) {
      throw std::length_error(
          "unsafeSetSize: new_size exceeds capacity. Use reserve() first.");
    }
    m_size = new_size;
  }

  void reserve(size_t new_capacity, cudaStream_t stream = 0) {
    if (new_capacity <= m_capacity) {
      return;
    }
    reallocate(new_capacity, true, stream);
  }

  void resize(size_t new_size, bool preserve_data = true,
              cudaStream_t stream = 0) {
    if (new_size == m_size) {
      return;
    }

    if (new_size == 0) {
      m_size = 0;
      return;
    }

    if (new_size <= m_capacity) {
      m_size = new_size;
      return;
    }

    reallocate(new_size, preserve_data, stream);
    m_size = new_size;
  }

  void clear() { m_size = 0; }

  void shrinkToFit(cudaStream_t stream = 0) {
    if (m_capacity == m_size) {
      return;
    }

    if (m_size == 0) {
      freeMemory();
      m_capacity = 0;
      return;
    }

    T *new_ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&new_ptr, m_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(new_ptr, m_ptr, m_size * sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream));

    freeMemory();
    m_ptr = new_ptr;
    m_capacity = m_size;
  }

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

  void clearAsync(int byte_value = 0, cudaStream_t stream = 0) {
    if (m_size == 0 || !m_ptr)
      return;
    CHECK_CUDA_ERROR(cudaMemsetAsync(m_ptr, byte_value, bytes(), stream));
  }

  void clearRangeAsync(size_t offset, size_t count, int byte_value = 0,
                       cudaStream_t stream = 0) {
    if (count == 0)
      return;
    if (offset + count > m_size) {
      throw std::out_of_range("clearRangeAsync: range exceeds valid size");
    }
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(m_ptr + offset, byte_value, count * sizeof(T), stream));
  }


  void initFromHost(const T *src_ptr, size_t count, cudaStream_t stream = 0) {
    if (count == 0) {
      m_size = 0;
      return;
    }

    if (count > m_capacity) {
      reallocate(count, false, stream);
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_ptr, src_ptr, count * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    m_size = count;
  }

  void initFromHost(const std::vector<T> &src, cudaStream_t stream = 0) {
    initFromHost(src.data(), src.size(), stream);
  }

  void initFromHost(const CudaHostBuffer<T> &src, cudaStream_t stream = 0) {
    initFromHost(src.readPtr(), src.size(), stream);
  }


  void writeFromHost(const T *src_ptr, size_t dst_offset, size_t count,
                     cudaStream_t stream = 0) {
    if (count == 0)
      return;

    size_t required_size = dst_offset + count;
    if (required_size > m_capacity) {
      throw std::out_of_range(
          "writeFromHost: exceeds capacity. Call reserve() first.");
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_ptr + dst_offset, src_ptr,
                                     count * sizeof(T), cudaMemcpyHostToDevice,
                                     stream));
    m_size = std::max(m_size, required_size);
  }

  // Device-to-host transfers enqueue on `stream` unless stated otherwise.

  void readToHost(T *dst_ptr, cudaStream_t stream = 0) const {
    if (m_size == 0)
      return;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst_ptr, m_ptr, m_size * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream));
  }

  void readToHost(T *dst_ptr, size_t src_offset, size_t count,
                  cudaStream_t stream = 0) const {
    if (count == 0)
      return;
    if (src_offset + count > m_size) {
      throw std::out_of_range("readToHost: range exceeds valid size");
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst_ptr, m_ptr + src_offset,
                                     count * sizeof(T), cudaMemcpyDeviceToHost,
                                     stream));
  }

  std::vector<T> toVector(cudaStream_t stream = 0) const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      readToHost(result.data(), stream);
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    return result;
  }

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

  // Device-to-device transfers preserve source data and update destination size.

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

  void writeFromDevice(const CudaDeviceBuffer<T> &src, size_t src_offset,
                       size_t dst_offset, size_t count,
                       cudaStream_t stream = 0) {
    if (count == 0)
      return;

    if (src_offset + count > src.size()) {
      throw std::out_of_range("writeFromDevice: source range out of bounds");
    }

    size_t required_size = dst_offset + count;
    if (required_size > m_capacity) {
      throw std::out_of_range(
          "writeFromDevice: exceeds capacity. Call reserve() first.");
    }

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(m_ptr + dst_offset, src.readPtr() + src_offset,
                        count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    m_size = std::max(m_size, required_size);
  }

private:
  void prepareForWrite(size_t count, cudaStream_t stream) {
    if (count > m_capacity) {
      // The caller will overwrite the entire valid range.
      reallocate(count, false, stream);
    }
    m_size = count;
  }

  void reallocate(size_t new_capacity, bool preserve_data,
                  cudaStream_t stream) {
    T *new_ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&new_ptr, new_capacity * sizeof(T)));

    if (preserve_data && m_ptr && m_size > 0) {
      size_t copy_count = std::min(m_size, new_capacity);
      CHECK_CUDA_ERROR(cudaMemcpyAsync(new_ptr, m_ptr, copy_count * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream));
    }

    freeMemory();
    m_ptr = new_ptr;
    m_capacity = new_capacity;
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

// Enables unqualified `swap` without exposing allocation details.
template <typename T>
void swap(CudaDeviceBuffer<T> &a, CudaDeviceBuffer<T> &b) noexcept {
  a.swap(b);
}

using DeviceByteBuffer = CudaDeviceBuffer<uint8_t>;

} // namespace ai_core::cuda_utils

#endif // CUDA_UTILS_CUDA_DEVICE_BUFFER_HPP
