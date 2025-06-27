/**
 * @file postproc_registrar.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __POSTPROC_REGISTRAR_HPP__
#define __POSTPROC_REGISTRAR_HPP__

#include "postproc_base.hpp"
#include "type_safe_factory.hpp"

namespace ai_core::dnn {

using PostprocFactory = common_utils::Factory<PostprocssBase>;

class PostprocessRegistrar {
public:
  static PostprocessRegistrar &getInstance() {
    static PostprocessRegistrar instance;
    return instance;
  }

  PostprocessRegistrar(const PostprocessRegistrar &) = delete;
  PostprocessRegistrar &operator=(const PostprocessRegistrar &) = delete;
  PostprocessRegistrar(PostprocessRegistrar &&) = delete;
  PostprocessRegistrar &operator=(PostprocessRegistrar &&) = delete;

private:
  PostprocessRegistrar();
};

[[maybe_unused]] inline const static PostprocessRegistrar &postproc_registrar =
    PostprocessRegistrar::getInstance();
} // namespace ai_core::dnn

#endif