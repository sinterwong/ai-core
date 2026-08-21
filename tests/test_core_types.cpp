#include "ai_core/algo_types.hpp"
#include "ai_core/data_packet.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/param_center.hpp"
#include "ai_core/type_safe_factory.hpp"
#include "gtest/gtest.h"

#include <sstream>
#include <string_view>

namespace testing_core_types {
using namespace ai_core;

// DataPacket

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

// ParamCenter

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

// AlgoOutput's trailing DataPacket is the extension slot for out-of-tree
// postprocessors: a plugin can return a result type this variant has never
// heard of, still typed, without editing the header.
namespace {
struct PoseRet {
  std::vector<Point2f> keypoints;
  float score;
};
} // namespace

TEST(AlgoOutputTest, DataPacketCarriesCustomPluginResults) {
  PoseRet pose;
  pose.keypoints = {{1.f, 2.f}, {3.f, 4.f}};
  pose.score = 0.9f;

  DataPacket packet;
  packet.setParam("pose", pose);

  AlgoOutput output;
  output.setParams(packet);

  const auto *held = output.getParams<DataPacket>();
  ASSERT_NE(held, nullptr);
  const auto restored = held->getParam<PoseRet>("pose");
  ASSERT_EQ(restored.keypoints.size(), 2u);
  EXPECT_FLOAT_EQ(restored.keypoints[1].x, 3.f);
  EXPECT_FLOAT_EQ(restored.score, 0.9f);

  // Still a variant: the built-in alternatives are unaffected.
  EXPECT_EQ(output.getParams<ClsRet>(), nullptr);
}

TEST(AlgoOutputTest, BuiltinAlternativesStillWork) {
  AlgoOutput output;
  ClsRet cls;
  cls.label = 3;
  cls.score = 0.5f;
  output.setParams(cls);

  ASSERT_NE(output.getParams<ClsRet>(), nullptr);
  EXPECT_EQ(output.getParams<ClsRet>()->label, 3);
  EXPECT_EQ(output.getParams<DataPacket>(), nullptr);
}

// InferErrorCode::to_string

TEST(ErrorCodeTest, ToStringNamesCodes) {
  struct ExpectedName {
    InferErrorCode code;
    std::string_view name;
  };
  constexpr ExpectedName expected_names[] = {
      {InferErrorCode::SUCCESS, "SUCCESS"},
      {InferErrorCode::InitFailed, "InitFailed"},
      {InferErrorCode::InitConfigFailed, "InitConfigFailed"},
      {InferErrorCode::InitModelLoadFailed, "InitModelLoadFailed"},
      {InferErrorCode::InitDeviceFailed, "InitDeviceFailed"},
      {InferErrorCode::InitMemoryAllocFailed, "InitMemoryAllocFailed"},
      {InferErrorCode::InitDecryptionFailed, "InitDecryptionFailed"},
      {InferErrorCode::NotInitialized, "NotInitialized"},
      {InferErrorCode::InitRuntimeFailed, "InitRuntimeFailed"},
      {InferErrorCode::InitEngineFailed, "InitEngineFailed"},
      {InferErrorCode::InitContextFailed, "InitContextFailed"},
      {InferErrorCode::InitBindingFailed, "InitBindingFailed"},
      {InferErrorCode::InferFailed, "InferFailed"},
      {InferErrorCode::InferInputError, "InferInputError"},
      {InferErrorCode::InferOutputError, "InferOutputError"},
      {InferErrorCode::InferDeviceError, "InferDeviceError"},
      {InferErrorCode::InferPreprocessFailed, "InferPreprocessFailed"},
      {InferErrorCode::InferMemoryError, "InferMemoryError"},
      {InferErrorCode::InferSetInputFailed, "InferSetInputFailed"},
      {InferErrorCode::InferExtractFailed, "InferExtractFailed"},
      {InferErrorCode::InferUnsupportedOutputType,
       "InferUnsupportedOutputType"},
      {InferErrorCode::InferTypeMismatch, "InferTypeMismatch"},
      {InferErrorCode::InferSizeMismatch, "InferSizeMismatch"},
      {InferErrorCode::InferInvalidInput, "InferInvalidInput"},
      {InferErrorCode::InferExecutionFailed, "InferExecutionFailed"},
      {InferErrorCode::InferBindingError, "InferBindingError"},
      {InferErrorCode::StreamCreationFailed, "StreamCreationFailed"},
      {InferErrorCode::StreamSyncFailed, "StreamSyncFailed"},
      {InferErrorCode::GraphCaptureFailed, "GraphCaptureFailed"},
      {InferErrorCode::GraphLaunchFailed, "GraphLaunchFailed"},
      {InferErrorCode::AsyncOperationPending, "AsyncOperationPending"},
      {InferErrorCode::TerminateFailed, "TerminateFailed"},
      {InferErrorCode::AlgoNotFound, "AlgoNotFound"},
      {InferErrorCode::AlgoRegisterFailed, "AlgoRegisterFailed"},
      {InferErrorCode::AlgoUnregisterFailed, "AlgoUnregisterFailed"},
      {InferErrorCode::AlgoInferFailed, "AlgoInferFailed"},
  };

  for (const auto &[code, name] : expected_names) {
    EXPECT_EQ(to_string(code), name);
  }
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

// Factory

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
