#ifndef AI_CORE_MODEL_OUTPUT_HPP
#define AI_CORE_MODEL_OUTPUT_HPP

#include <map>
#include <string>
#include <vector>

#include "ai_core/typed_buffer.hpp"

namespace ai_core {

struct TensorData {
  std::map<std::string, TypedBuffer> datas;
  std::map<std::string, std::vector<int>> shapes;
};

} // namespace ai_core
#endif
