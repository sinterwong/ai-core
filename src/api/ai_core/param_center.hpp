/**
 * @file param_center.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_PARAM_CENTER_HPP
#define AI_CORE_PARAM_CENTER_HPP

#include <algorithm>
#include <variant>

namespace ai_core {
template <typename P> class ParamCenter {
public:
  using Params = P;
  template <typename T> void setParams(T params) {
    m_params = std::move(params);
  }

  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               m_params);
  }

  template <typename Func> void visitParams(Func &&func) const {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               m_params);
  }

  template <typename T> T *getParams() { return std::get_if<T>(&m_params); }

  template <typename T> const T *getParams() const {
    return std::get_if<T>(&m_params);
  }

private:
  Params m_params;
};
} // namespace ai_core

#endif