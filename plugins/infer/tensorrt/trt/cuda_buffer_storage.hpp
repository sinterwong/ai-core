#ifndef AI_CORE_CUDA_BUFFER_STORAGE_HPP
#define AI_CORE_CUDA_BUFFER_STORAGE_HPP

#include "ai_core/typed_buffer.hpp"

namespace ai_core::dnn::gpu {

TypedBuffer allocateCudaDeviceBuffer(DataType type, size_t size_bytes,
                                     int device_id = 0);
TypedBuffer allocateCudaPinnedBuffer(DataType type, size_t size_bytes,
                                     int device_id = 0);

} // namespace ai_core::dnn::gpu

#endif
