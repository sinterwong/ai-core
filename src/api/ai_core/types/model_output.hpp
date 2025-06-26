#ifndef __AI_CORE_MODEL_OUTPUT_HPP__
#define __AI_CORE_MODEL_OUTPUT_HPP__

#include <map>
#include <string>
#include <vector>
#include "typed_buffer.hpp" // Assuming this will be in the same directory

namespace ai_core {

// Model output(after infering, before postprocess)
struct ModelOutput {
  std::map<std::string, TypedBuffer> outputs;
  std::map<std::string, std::vector<int>> outputShapes;
};

} // namespace ai_core
#endif // __AI_CORE_MODEL_OUTPUT_HPP__
