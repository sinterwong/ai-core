/**
 * @file test_typed_buffer.cpp
 * @brief Unit tests for TypedBuffer: factory semantics, copy/move, resize
 * split, type-checked access. No model assets required.
 */
#include "ai_core/typed_buffer.hpp"
#include "gtest/gtest.h"

#include <cstring>
#include <numeric>
#include <vector>

namespace testing_typed_buffer {
using namespace ai_core;

std::vector<uint8_t> floatBytes(const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

TEST(TypedBufferTest, DefaultConstructedIsEmpty) {
  TypedBuffer buf;
  EXPECT_EQ(buf.getElementCount(), 0u);
  EXPECT_EQ(buf.getSizeBytes(), 0u);
  EXPECT_EQ(buf.dataType(), DataType::FLOAT32);
  EXPECT_EQ(buf.location(), BufferLocation::CPU);
  EXPECT_EQ(buf.memoryType(), BufferMemoryType::Pageable);
  EXPECT_FALSE(buf.isReference());
  EXPECT_FALSE(buf.isPinned());
}

TEST(TypedBufferTest, GetElementSize) {
  EXPECT_EQ(TypedBuffer::getElementSize(DataType::FLOAT32), 4u);
  EXPECT_EQ(TypedBuffer::getElementSize(DataType::FLOAT16), 2u);
  EXPECT_EQ(TypedBuffer::getElementSize(DataType::INT32), 4u);
  EXPECT_EQ(TypedBuffer::getElementSize(DataType::INT64), 8u);
  EXPECT_EQ(TypedBuffer::getElementSize(DataType::INT8), 1u);
}

TEST(TypedBufferTest, CreateFromCpuCopiesAndCountsElements) {
  const std::vector<float> values{1.f, 2.f, 3.f};
  auto bytes = floatBytes(values);
  TypedBuffer buf = TypedBuffer::createFromCpu(DataType::FLOAT32, bytes);

  EXPECT_EQ(buf.getElementCount(), 3u);
  EXPECT_EQ(buf.getSizeBytes(), 12u);
  EXPECT_NE(buf.getRawHostPtr(), bytes.data()); // owns its copy

  const float *data = buf.getHostPtr<float>();
  EXPECT_EQ(data[0], 1.f);
  EXPECT_EQ(data[2], 3.f);
}

TEST(TypedBufferTest, CreateFromCpuMove) {
  TypedBuffer buf =
      TypedBuffer::createFromCpu(DataType::INT64, floatBytes({1.f, 2.f}));
  // 8 bytes of input / 8 bytes per int64 element
  EXPECT_EQ(buf.getElementCount(), 1u);
  EXPECT_EQ(buf.dataType(), DataType::INT64);
}

TEST(TypedBufferTest, WrapCpuIsNonOwningReference) {
  std::vector<float> values{5.f, 6.f, 7.f, 8.f};
  TypedBuffer buf = TypedBuffer::wrapCpu(DataType::FLOAT32, values.data(),
                                         values.size() * sizeof(float));

  EXPECT_TRUE(buf.isReference());
  EXPECT_EQ(buf.getElementCount(), 4u);
  EXPECT_EQ(buf.getRawHostPtr(), values.data()); // zero copy

  // Mutating the original is visible through the view
  values[0] = 50.f;
  EXPECT_EQ(buf.getHostPtr<float>()[0], 50.f);
}

TEST(TypedBufferTest, CopyOfWrappedBufferIsDeep) {
  std::vector<float> values{1.f, 2.f};
  TypedBuffer view = TypedBuffer::wrapCpu(DataType::FLOAT32, values.data(),
                                          values.size() * sizeof(float));
  TypedBuffer copy = view;

  EXPECT_FALSE(copy.isReference());
  EXPECT_NE(copy.getRawHostPtr(), values.data());
  EXPECT_EQ(copy.getHostPtr<float>()[1], 2.f);

  // Deep copy detached from the original
  values[1] = 20.f;
  EXPECT_EQ(copy.getHostPtr<float>()[1], 2.f);
}

TEST(TypedBufferTest, MoveLeavesSourceEmpty) {
  TypedBuffer src =
      TypedBuffer::createFromCpu(DataType::FLOAT32, floatBytes({1.f, 2.f}));
  TypedBuffer dst = std::move(src);

  EXPECT_EQ(dst.getElementCount(), 2u);
  EXPECT_EQ(src.getElementCount(), 0u); // NOLINT(bugprone-use-after-move)
}

TEST(TypedBufferTest, CopyAssignReplacesContents) {
  TypedBuffer a =
      TypedBuffer::createFromCpu(DataType::FLOAT32, floatBytes({1.f}));
  TypedBuffer b =
      TypedBuffer::createFromCpu(DataType::FLOAT32, floatBytes({9.f, 8.f}));
  a = b;
  EXPECT_EQ(a.getElementCount(), 2u);
  EXPECT_EQ(a.getHostPtr<float>()[0], 9.f);
  // Independent storage
  EXPECT_NE(a.getRawHostPtr(), b.getRawHostPtr());
}

TEST(TypedBufferTest, TypeMismatchedHostAccessThrows) {
  TypedBuffer buf =
      TypedBuffer::createFromCpu(DataType::FLOAT32, floatBytes({1.f}));
  EXPECT_THROW(buf.getHostPtr<int64_t>(), std::runtime_error);
  EXPECT_NO_THROW(buf.getHostPtr<float>());
  EXPECT_THROW(buf.getRawDevicePtr(), std::runtime_error);
}

TEST(TypedBufferTest, ResizeDiscardAllocates) {
  TypedBuffer buf;
  buf.resizeDiscard(16);
  EXPECT_EQ(buf.getElementCount(), 16u);
  EXPECT_EQ(buf.getSizeBytes(), 64u); // FLOAT32 default
  EXPECT_NE(buf.getRawHostPtr(), nullptr);
}

TEST(TypedBufferTest, ResizeDiscardDetachesWrappedMemory) {
  std::vector<float> values{1.f, 2.f};
  TypedBuffer buf = TypedBuffer::wrapCpu(DataType::FLOAT32, values.data(),
                                         values.size() * sizeof(float));
  buf.resizeDiscard(4);
  EXPECT_FALSE(buf.isReference());
  EXPECT_NE(buf.getRawHostPtr(), values.data());
  EXPECT_EQ(buf.getElementCount(), 4u);
}

TEST(TypedBufferTest, ResizePreservingKeepsContents) {
  TypedBuffer buf =
      TypedBuffer::createFromCpu(DataType::FLOAT32, floatBytes({1.f, 2.f}));
  buf.resizePreserving(4);
  EXPECT_EQ(buf.getElementCount(), 4u);
  EXPECT_EQ(buf.getHostPtr<float>()[0], 1.f);
  EXPECT_EQ(buf.getHostPtr<float>()[1], 2.f);

  buf.resizePreserving(1);
  EXPECT_EQ(buf.getElementCount(), 1u);
  EXPECT_EQ(buf.getHostPtr<float>()[0], 1.f);
}

TEST(TypedBufferTest, ResizePreservingDetachesWrappedMemory) {
  std::vector<float> values{3.f, 4.f};
  TypedBuffer buf = TypedBuffer::wrapCpu(DataType::FLOAT32, values.data(),
                                         values.size() * sizeof(float));
  buf.resizePreserving(3);
  EXPECT_FALSE(buf.isReference());
  EXPECT_EQ(buf.getElementCount(), 3u);
  EXPECT_EQ(buf.getHostPtr<float>()[0], 3.f);
  EXPECT_EQ(buf.getHostPtr<float>()[1], 4.f);
}

TEST(TypedBufferTest, ClearResetsState) {
  TypedBuffer buf = TypedBuffer::createFromCpu(DataType::INT8, {1, 2, 3});
  buf.clear();
  EXPECT_EQ(buf.getElementCount(), 0u);
  EXPECT_EQ(buf.getSizeBytes(), 0u);
  EXPECT_EQ(buf.dataType(), DataType::FLOAT32);
}

#ifdef WITH_TRT
TEST(TypedBufferTest, PinnedHostAllocationAndResizeContract) {
  TypedBuffer buf = TypedBuffer::createPinnedHost(DataType::FLOAT32, 64);
  EXPECT_TRUE(buf.isPinned());
  EXPECT_EQ(buf.location(), BufferLocation::CPU);
  EXPECT_EQ(buf.getElementCount(), 16u);
  EXPECT_NE(buf.getRawHostPtr(), nullptr);

  // Preserving resize is CPU-pageable-only by contract
  EXPECT_THROW(buf.resizePreserving(32), std::logic_error);
  EXPECT_NO_THROW(buf.resizeDiscard(32));
  EXPECT_EQ(buf.getElementCount(), 32u);
}

TEST(TypedBufferTest, GpuAllocateAndWrap) {
  TypedBuffer gpu = TypedBuffer::allocateGpu(DataType::FLOAT32, 64);
  EXPECT_EQ(gpu.location(), BufferLocation::GpuDevice);
  EXPECT_EQ(gpu.getElementCount(), 16u);
  ASSERT_NE(gpu.getRawDevicePtr(), nullptr);
  EXPECT_THROW(gpu.getRawHostPtr(), std::runtime_error);
  EXPECT_THROW(gpu.resizePreserving(32), std::logic_error);

  TypedBuffer wrapped =
      TypedBuffer::wrapGpu(DataType::FLOAT32, gpu.getRawDevicePtr(), 64);
  EXPECT_EQ(wrapped.getRawDevicePtr(), gpu.getRawDevicePtr());
  EXPECT_EQ(wrapped.getElementCount(), 16u);
}
#endif

} // namespace testing_typed_buffer
