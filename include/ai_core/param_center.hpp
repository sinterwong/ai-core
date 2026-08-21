#ifndef AI_CORE_PARAM_CENTER_HPP
#define AI_CORE_PARAM_CENTER_HPP

#include <algorithm>
#include <variant>

namespace ai_core {
/**
 * @brief Type-safe variant wrapper for algorithm inputs/outputs/params.
 *
 * @par Thread safety
 * Value type with no internal synchronization; concurrent const access is
 * safe, concurrent mutation requires external synchronization.
 */
template <typename P> class ParamCenter {
public:
  using Params = P;

  /** Replace the active variant alternative with `params`. */
  template <typename T> void setParams(T params) {
    m_params = std::move(params);
  }

  /** Invoke `func` with the active alternative. */
  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               m_params);
  }

  /** Invoke `func` with the active alternative as a const value. */
  template <typename Func> void visitParams(Func &&func) const {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               m_params);
  }

  /** Return the active value when it has type `T`, otherwise `nullptr`. */
  template <typename T> T *getParams() { return std::get_if<T>(&m_params); }

  template <typename T> const T *getParams() const {
    return std::get_if<T>(&m_params);
  }

private:
  Params m_params;
};
} // namespace ai_core

#endif
