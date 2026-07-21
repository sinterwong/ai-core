/**
 * @file test_core_types.cpp
 * @brief Unit tests for DataPacket, ParamCenter and the type-safe Factory.
 * No model assets required.
 */
#include "ai_core/algo_types.hpp"
#include "ai_core/data_packet.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/param_center.hpp"
#include "ai_core/type_safe_factory.hpp"
#include "gtest/gtest.h"

#include <sstream>

namespace testing_core_types {
using namespace ai_core;

// ============================================================================
// DataPacket
// ============================================================================

TEST(DataPacketTest, SetAndGet) {
  DataPacket packet;
  packet.setParam("count", 42);
  packet.setParam("name", std::string("yolo"));

  EXPECT_EQ(packet.getParam<int>("count"), 42);
  EXPECT_EQ(packet.getParam<std::string>("name"), "yolo");
}

TEST(DataPacketTest, MissingKeyThrows) {
  DataPacket packet;
  EXPECT_THROW(packet.getParam<int>("missing"), std::runtime_error);
}

TEST(DataPacketTest, WrongTypeThrows) {
  DataPacket packet;
  packet.setParam("count", 42);
  EXPECT_THROW(packet.getParam<std::string>("count"), std::runtime_error);
}

TEST(DataPacketTest, OptionalParam) {
  DataPacket packet;
  packet.setParam("present", 1.5);

  EXPECT_EQ(packet.getOptionalParam<double>("present"), 1.5);
  EXPECT_EQ(packet.getOptionalParam<double>("absent"), std::nullopt);
  // Present but wrong type is an error, not nullopt
  EXPECT_THROW(packet.getOptionalParam<int>("present"), std::runtime_error);
}

TEST(DataPacketTest, HasVariants) {
  DataPacket packet;
  packet.setParam("key", 7);

  EXPECT_TRUE(packet.has("key"));
  EXPECT_FALSE(packet.has("other"));
  EXPECT_TRUE(packet.has<int>("key"));
  EXPECT_FALSE(packet.has<std::string>("key"));
  EXPECT_TRUE(packet.has<int>());
  EXPECT_FALSE(packet.has<float>());
}

TEST(DataPacketTest, SetOverwrites) {
  DataPacket packet;
  packet.setParam("key", 1);
  packet.setParam("key", std::string("two"));
  EXPECT_EQ(packet.getParam<std::string>("key"), "two");
  EXPECT_THROW(packet.getParam<int>("key"), std::runtime_error);
}

// ============================================================================
// ParamCenter
// ============================================================================

TEST(ParamCenterTest, DefaultHoldsMonostate) {
  AlgoPreprocParams params;
  EXPECT_EQ(params.getParams<FramePreprocessArg>(), nullptr);
}

TEST(ParamCenterTest, SetGetRoundTrip) {
  AlgoPreprocParams params;
  FramePreprocessArg arg;
  arg.model_input_shape = {640, 480, 3};
  params.setParams(arg);

  const auto *stored = params.getParams<FramePreprocessArg>();
  ASSERT_NE(stored, nullptr);
  EXPECT_EQ(stored->model_input_shape.w, 640);
  EXPECT_EQ(stored->model_input_shape.h, 480);
}

TEST(ParamCenterTest, WrongTypeReturnsNull) {
  AlgoPostprocParams params;
  AnchorDetParams anchor;
  params.setParams(anchor);
  EXPECT_NE(params.getParams<AnchorDetParams>(), nullptr);
  EXPECT_EQ(params.getParams<GenericPostParams>(), nullptr);
}

TEST(ParamCenterTest, VisitDispatchesToHeldAlternative) {
  AlgoPostprocParams params;
  AnchorDetParams anchor;
  anchor.cond_thre = 0.25f;
  params.setParams(anchor);

  bool visited_anchor = false;
  params.visitParams([&](const auto &held) {
    using T = std::decay_t<decltype(held)>;
    if constexpr (std::is_same_v<T, AnchorDetParams>) {
      visited_anchor = true;
      EXPECT_FLOAT_EQ(held.cond_thre, 0.25f);
    }
  });
  EXPECT_TRUE(visited_anchor);
}

// ============================================================================
// InferErrorCode::to_string
// ============================================================================

TEST(ErrorCodeTest, ToStringNamesCodes) {
  EXPECT_EQ(to_string(InferErrorCode::SUCCESS), "SUCCESS");
  EXPECT_EQ(to_string(InferErrorCode::InferSizeMismatch), "InferSizeMismatch");
  EXPECT_EQ(to_string(InferErrorCode::AlgoNotFound), "AlgoNotFound");
}

TEST(ErrorCodeTest, UnknownValueIsHandled) {
  EXPECT_EQ(to_string(static_cast<InferErrorCode>(99999)),
            "InferErrorCode(unknown)");
}

TEST(ErrorCodeTest, StreamOperatorIncludesNumericValue) {
  std::ostringstream oss;
  oss << InferErrorCode::NotInitialized;
  EXPECT_EQ(oss.str(), "NotInitialized(106)");
}

// ============================================================================
// Factory
// ============================================================================

struct TestBase {
  virtual ~TestBase() = default;
  virtual int id() const = 0;
};

struct ImplA : TestBase {
  int id() const override { return 1; }
};

struct ImplWithParams : TestBase {
  explicit ImplWithParams(int v) : value(v) {}
  int id() const override { return value; }
  int value;
};

TEST(FactoryTest, RegisterAndCreate) {
  auto &factory = Factory<TestBase>::instance();
  factory.registerCreator(
      "ImplA", [](const DataPacket &) { return std::make_shared<ImplA>(); });

  EXPECT_TRUE(factory.isRegistered("ImplA"));
  auto obj = factory.create("ImplA");
  ASSERT_NE(obj, nullptr);
  EXPECT_EQ(obj->id(), 1);
}

TEST(FactoryTest, CreatePassesParams) {
  auto &factory = Factory<TestBase>::instance();
  factory.registerCreator("ImplWithParams", [](const DataPacket &params) {
    return std::make_shared<ImplWithParams>(params.getParam<int>("value"));
  });

  DataPacket params;
  params.setParam("value", 77);
  auto obj = factory.create("ImplWithParams", params);
  EXPECT_EQ(obj->id(), 77);
}

TEST(FactoryTest, UnknownNameThrows) {
  auto &factory = Factory<TestBase>::instance();
  EXPECT_FALSE(factory.isRegistered("Nope"));
  EXPECT_THROW(factory.create("Nope"), std::runtime_error);
}

TEST(FactoryTest, DuplicateRegistrationRejected) {
  auto &factory = Factory<TestBase>::instance();
  factory.registerCreator(
      "Dup", [](const DataPacket &) { return std::make_shared<ImplA>(); });
  bool second = factory.registerCreator(
      "Dup", [](const DataPacket &) { return std::make_shared<ImplA>(); });
  EXPECT_FALSE(second);
}

TEST(FactoryTest, NullCreatorThrows) {
  auto &factory = Factory<TestBase>::instance();
  EXPECT_THROW(factory.registerCreator("Null", nullptr), std::runtime_error);
}

TEST(FactoryTest, CreatorExceptionIsWrapped) {
  auto &factory = Factory<TestBase>::instance();
  factory.registerCreator("Throws",
                          [](const DataPacket &) -> std::shared_ptr<TestBase> {
                            throw std::invalid_argument("boom");
                          });
  EXPECT_THROW(factory.create("Throws"), std::runtime_error);
}

} // namespace testing_core_types
