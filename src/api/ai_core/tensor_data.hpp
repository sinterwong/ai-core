/**
 * @file tensor_data.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Named tensor aggregate exchanged between pipeline stages.
 * @version 0.2
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_MODEL_OUTPUT_HPP
#define AI_CORE_MODEL_OUTPUT_HPP

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ai_core/typed_buffer.hpp"

namespace ai_core {

/**
 * @brief One named tensor: buffer + shape. The element dtype lives on the
 * buffer (TypedBuffer::dataType()).
 */
struct Tensor {
  std::string name;
  TypedBuffer buffer;
  std::vector<int> shape;
};

/**
 * @brief Ordered collection of named tensors.
 *
 * Models have 1~3 inputs/outputs, so tensors are kept in a flat vector in
 * insertion order and looked up by linear scan — cheaper than a map at these
 * sizes, and iteration order is deterministic.
 *
 * @par Thread safety
 * Value type with no internal synchronization; concurrent const access is
 * safe, concurrent mutation requires external synchronization.
 */
class TensorData {
public:
  using const_iterator = std::vector<Tensor>::const_iterator;
  using iterator = std::vector<Tensor>::iterator;

  /**
   * @brief Insert a tensor, replacing any existing tensor of the same name.
   */
  Tensor &set(std::string name, TypedBuffer buffer, std::vector<int> shape) {
    if (Tensor *existing = find(name)) {
      existing->buffer = std::move(buffer);
      existing->shape = std::move(shape);
      return *existing;
    }
    m_tensors.push_back(
        Tensor{std::move(name), std::move(buffer), std::move(shape)});
    return m_tensors.back();
  }

  Tensor &set(Tensor tensor) {
    return set(std::move(tensor.name), std::move(tensor.buffer),
               std::move(tensor.shape));
  }

  const Tensor *find(std::string_view name) const noexcept {
    for (const auto &t : m_tensors) {
      if (t.name == name) {
        return &t;
      }
    }
    return nullptr;
  }

  Tensor *find(std::string_view name) noexcept {
    return const_cast<Tensor *>(
        static_cast<const TensorData *>(this)->find(name));
  }

  const Tensor &at(std::string_view name) const {
    if (const Tensor *t = find(name)) {
      return *t;
    }
    throw std::out_of_range("TensorData: no tensor named '" +
                            std::string(name) + "'");
  }

  bool contains(std::string_view name) const noexcept {
    return find(name) != nullptr;
  }

  size_t size() const noexcept { return m_tensors.size(); }
  bool empty() const noexcept { return m_tensors.empty(); }
  void clear() noexcept { m_tensors.clear(); }

  const_iterator begin() const noexcept { return m_tensors.begin(); }
  const_iterator end() const noexcept { return m_tensors.end(); }
  iterator begin() noexcept { return m_tensors.begin(); }
  iterator end() noexcept { return m_tensors.end(); }

private:
  std::vector<Tensor> m_tensors;
};

} // namespace ai_core
#endif
