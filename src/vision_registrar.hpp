/**
 * @file vision_registrar.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __VISION_REGISTRAR_HPP__
#define __VISION_REGISTRAR_HPP__

#include "type_safe_factory.hpp"
#include "vision.hpp"

namespace ai_core::dnn::vision {

using VisionFactory = common_utils::Factory<VisionBase>;

class VisionRegistrar {
public:
  static VisionRegistrar &getInstance() {
    static VisionRegistrar instance;
    return instance;
  }

  VisionRegistrar(const VisionRegistrar &) = delete;
  VisionRegistrar &operator=(const VisionRegistrar &) = delete;
  VisionRegistrar(VisionRegistrar &&) = delete;
  VisionRegistrar &operator=(VisionRegistrar &&) = delete;

private:
  VisionRegistrar();
};

[[maybe_unused]] inline const static VisionRegistrar &node_registrar =
    VisionRegistrar::getInstance();
} // namespace ai_core::dnn::vision

#endif