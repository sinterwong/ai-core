#ifndef __AI_CORE_TYPES_H__
#define __AI_CORE_TYPES_H__
#include "ai_export.h"
#include <cstdint>
#include <string>
#include <vector>

namespace ai_core {

enum class ErrorCode {
  SUCCESS = 0,
  INVALID_INPUT = -1,
  FILE_NOT_FOUND = -2,
  INVALID_FILE_FORMAT = -3,
  INITIALIZATION_FAILED = -4,
  PROCESSING_ERROR = -5,
  INVALID_STATE = -6,
  TRY_GET_NEXT_OVERTIME = -7
};
} // namespace ai_core
#endif
