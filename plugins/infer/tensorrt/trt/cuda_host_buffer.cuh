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

template <typename T> class CudaDeviceBuffer;

/** Non-owning read-only range over pinned host memory. */
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

/** Non-owning writable range over pinned host memory. */
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
 * @brief Move-only vector-like container backed by CUDA pinned host memory.
 *
 * Pinned allocation makes host/device copies eligible for asynchronous DMA.
 * Any buffer participating in an asynchronous copy must remain alive until
 * the associated CUDA stream completes.
 */
template <typename T> class CudaHostBuffer {
public:

  CudaHostBuffer() : m_size(0), m_capacity(0), m_ptr(nullptr) {}

  explicit CudaHostBuffer(size_t size)
      : m_size(size), m_capacity(size), m_ptr(nullptr) {
    if (m_capacity > 0) {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_capacity * sizeof(T)));
      std::memset(m_ptr, 0, m_capacity * sizeof(T));
    }
  }

  CudaHostBuffer(std::initializer_list<T> init)
      : m_size(init.size()), m_capacity(init.size()), m_ptr(nullptr) {
    if (m_capacity > 0) {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_capacity * sizeof(T)));
      std::copy(init.begin(), init.end(), m_ptr);
    }
  }

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

  CudaHostBuffer(const CudaHostBuffer &) = delete;
  CudaHostBuffer &operator=(const CudaHostBuffer &) = delete;

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


  size_t size() const { return m_size; }
  size_t capacity() const { return m_capacity; }
  size_t bytes() const { return m_size * sizeof(T); }
  size_t capacityBytes() const { return m_capacity * sizeof(T); }
  bool empty() const { return m_size == 0; }
  explicit operator bool() const { return m_ptr != nullptr; }


  HostReadSpan<T> readSpan() const { return {m_ptr, m_size}; }

  HostReadSpan<T> readSpan(size_t offset, size_t count) const {
    if (offset + count > m_size) {
      throw std::out_of_range("readSpan: range exceeds valid size");
    }
    return {m_ptr + offset, count};
  }

  const T *readPtr() const { return m_ptr; }


  /** Reserve `count` writable elements and mark the whole range valid. */
  HostWriteSpan<T> writeSpan(size_t count) {
    prepareForWrite(count);
    return {m_ptr, count};
  }

  T *writePtr(size_t count) {
    prepareForWrite(count);
    return m_ptr;
  }

  HostWriteSpan<T> writeSpanAt(size_t offset, size_t count) {
    size_t required_size = offset + count;
    if (required_size > m_capacity) {
      reallocate(required_size, true);
    }
    m_size = std::max(m_size, required_size);
    return {m_ptr + offset, count};
  }


  HostWriteSpan<T> appendSpan(size_t count) {
    size_t offset = m_size;
    size_t new_size = m_size + count;

    if (new_size > m_capacity) {
      size_t new_capacity = std::max(m_capacity * 2, new_size);
      reallocate(new_capacity, true);
    }

    m_size = new_size;
    return {m_ptr + offset, count};
  }

  T *appendPtr(size_t count) { return appendSpan(count).ptr; }


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

  T &operator[](size_t index) { return m_ptr[index]; }
  const T &operator[](size_t index) const { return m_ptr[index]; }

  T &front() { return m_ptr[0]; }
  const T &front() const { return m_ptr[0]; }

  T &back() { return m_ptr[m_size - 1]; }
  const T &back() const { return m_ptr[m_size - 1]; }


  T *begin() { return m_ptr; }
  T *end() { return m_ptr + m_size; }
  const T *begin() const { return m_ptr; }
  const T *end() const { return m_ptr + m_size; }
  const T *cbegin() const { return m_ptr; }
  const T *cend() const { return m_ptr + m_size; }


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
      throw std::length_error("unsafeSetSize: new_size exceeds capacity");
    }
    m_size = new_size;
  }


  void reserve(size_t new_capacity) {
    if (new_capacity <= m_capacity) {
      return;
    }
    reallocate(new_capacity, true);
  }

  void resize(size_t new_size, bool preserve_data = true) {
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

    reallocate(new_size, preserve_data);
    m_size = new_size;
  }

  void resize(size_t new_size, const T &value) {
    size_t old_size = m_size;
    resize(new_size, true);

    if (new_size > old_size) {
      std::fill(m_ptr + old_size, m_ptr + new_size, value);
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

    T *new_ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMallocHost(&new_ptr, m_size * sizeof(T)));
    std::memcpy(new_ptr, m_ptr, m_size * sizeof(T));

    freeMemory();
    m_ptr = new_ptr;
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


  void pushBack(const T &value) {
    if (m_size >= m_capacity) {
      size_t new_capacity = m_capacity == 0 ? 16 : m_capacity * 2;
      reallocate(new_capacity, true);
    }
    m_ptr[m_size++] = value;
  }

  void pushBack(T &&value) {
    if (m_size >= m_capacity) {
      size_t new_capacity = m_capacity == 0 ? 16 : m_capacity * 2;
      reallocate(new_capacity, true);
    }
    m_ptr[m_size++] = std::move(value);
  }

  void popBack() {
    if (m_size > 0) {
      --m_size;
    }
  }

  template <typename... Args> T &emplaceBack(Args &&...args) {
    if (m_size >= m_capacity) {
      size_t new_capacity = m_capacity == 0 ? 16 : m_capacity * 2;
      reallocate(new_capacity, true);
    }
    new (&m_ptr[m_size]) T(std::forward<Args>(args)...);
    return m_ptr[m_size++];
  }


  void fill(const T &value) { std::fill(m_ptr, m_ptr + m_size, value); }

  void zero() {
    if (m_size > 0) {
      std::memset(m_ptr, 0, m_size * sizeof(T));
    }
  }


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

  std::vector<T> toVector() const {
    return std::vector<T>(m_ptr, m_ptr + m_size);
  }

  CudaHostBuffer &operator=(const std::vector<T> &src) {
    initFromVector(src);
    return *this;
  }


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

  void readFromDevice(const CudaDeviceBuffer<T> &src, size_t src_offset,
                      size_t dst_offset, size_t count,
                      cudaStream_t stream = 0) {
    if (count == 0)
      return;

    if (src_offset + count > src.size()) {
      throw std::out_of_range("readFromDevice: source range out of bounds");
    }

    size_t required_size = dst_offset + count;
    if (required_size > m_capacity) {
      throw std::out_of_range(
          "readFromDevice: exceeds capacity. Call reserve() first.");
    }

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(m_ptr + dst_offset, src.readPtr() + src_offset,
                        count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    m_size = std::max(m_size, required_size);
  }

  void writeToDevice(CudaDeviceBuffer<T> &dst, cudaStream_t stream = 0) const {
    dst.initFromHost(m_ptr, m_size, stream);
  }

  void writeToDeviceAsync(CudaDeviceBuffer<T> &dst, size_t src_offset,
                          size_t dst_offset, size_t count,
                          cudaStream_t stream = 0) const {
    if (count == 0)
      return;

    if (src_offset + count > m_size) {
      throw std::out_of_range("writeToDeviceAsync: source range out of bounds");
    }

    dst.writeFromHost(m_ptr + src_offset, dst_offset, count, stream);
  }

  void copyTo(std::vector<T> &dest) const {
    if (dest.size() != m_size) {
      dest.resize(m_size);
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

  void reallocate(size_t new_capacity, bool preserve_data) {
    T *new_ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMallocHost(&new_ptr, new_capacity * sizeof(T)));

    if (preserve_data && m_ptr && m_size > 0) {
      size_t copy_count = std::min(m_size, new_capacity);
      std::memcpy(new_ptr, m_ptr, copy_count * sizeof(T));
    }

    freeMemory();
    m_ptr = new_ptr;
    m_capacity = new_capacity;
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

// Enables unqualified `swap` without exposing allocation details.
template <typename T>
void swap(CudaHostBuffer<T> &a, CudaHostBuffer<T> &b) noexcept {
  a.swap(b);
}

} // namespace ai_core::cuda_utils

#endif // CUDA_UTILS_CUDA_HOST_BUFFER_HPP
