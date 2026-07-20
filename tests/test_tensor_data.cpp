/**
 * @file test_tensor_data.cpp
 * @brief Unit tests for Tensor / TensorData: insertion order, name lookup,
 * replace semantics. No model assets required.
 */
#include "ai_core/tensor_data.hpp"
#include "gtest/gtest.h"

#include <cstring>

namespace testing_tensor_data {
using namespace ai_core;

TypedBuffer makeBuffer(std::vector<float> values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return TypedBuffer::createFromCpu(DataType::FLOAT32, std::move(bytes));
}

TEST(TensorDataTest, EmptyByDefault) {
  TensorData data;
  EXPECT_TRUE(data.empty());
  EXPECT_EQ(data.size(), 0u);
  EXPECT_EQ(data.find("anything"), nullptr);
  EXPECT_FALSE(data.contains("anything"));
  EXPECT_THROW(data.at("anything"), std::out_of_range);
}

TEST(TensorDataTest, SetAndLookup) {
  TensorData data;
  data.set("images", makeBuffer({1.f, 2.f}), {1, 2});

  ASSERT_TRUE(data.contains("images"));
  const Tensor &t = data.at("images");
  EXPECT_EQ(t.name, "images");
  EXPECT_EQ(t.shape, (std::vector<int>{1, 2}));
  EXPECT_EQ(t.buffer.getElementCount(), 2u);
  EXPECT_EQ(t.buffer.getHostPtr<float>()[1], 2.f);
}

TEST(TensorDataTest, PreservesInsertionOrder) {
  TensorData data;
  data.set("zeta", makeBuffer({1.f}), {1});
  data.set("alpha", makeBuffer({2.f}), {1});
  data.set("mid", makeBuffer({3.f}), {1});

  std::vector<std::string> names;
  for (const auto &t : data) {
    names.push_back(t.name);
  }
  EXPECT_EQ(names, (std::vector<std::string>{"zeta", "alpha", "mid"}));
}

TEST(TensorDataTest, SetReplacesExistingKeepingPosition) {
  TensorData data;
  data.set("a", makeBuffer({1.f}), {1});
  data.set("b", makeBuffer({2.f}), {1});
  data.set("a", makeBuffer({9.f, 8.f}), {2});

  EXPECT_EQ(data.size(), 2u);
  EXPECT_EQ(data.at("a").shape, (std::vector<int>{2}));
  EXPECT_EQ(data.at("a").buffer.getHostPtr<float>()[0], 9.f);
  EXPECT_EQ(data.begin()->name, "a"); // position kept
}

TEST(TensorDataTest, SetWholeTensor) {
  TensorData data;
  Tensor t{"logits", makeBuffer({0.5f}), {1, 1}};
  data.set(std::move(t));
  EXPECT_TRUE(data.contains("logits"));
  EXPECT_EQ(data.at("logits").shape, (std::vector<int>{1, 1}));
}

TEST(TensorDataTest, MutableFind) {
  TensorData data;
  data.set("x", makeBuffer({1.f}), {1});
  Tensor *t = data.find("x");
  ASSERT_NE(t, nullptr);
  t->shape = {4};
  EXPECT_EQ(data.at("x").shape, (std::vector<int>{4}));
}

TEST(TensorDataTest, ClearEmpties) {
  TensorData data;
  data.set("x", makeBuffer({1.f}), {1});
  data.clear();
  EXPECT_TRUE(data.empty());
  EXPECT_EQ(data.find("x"), nullptr);
}

} // namespace testing_tensor_data
