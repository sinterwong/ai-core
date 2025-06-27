#ifndef __AI_CORE_MODEL_OUTPUT_HPP__
#define __AI_CORE_MODEL_OUTPUT_HPP__

#include <map>
#include <string>
#include <vector>

#include "typed_buffer.hpp"

namespace ai_core {

struct TensorData {
  std::map<std::string, TypedBuffer> datas;
  std::map<std::string, std::vector<int>> shapes;
};

} // namespace ai_core
#endif // __AI_CORE_MODEL_OUTPUT_HPP__
