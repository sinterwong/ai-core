/**
 * @file test_preproc_cpu.cpp
 * @brief Unit tests for the CPU frame preprocessor (layout conversion,
 * normalization, ROI crop) and the parameter binding contract of the
 * AlgoPreproc facade. Synthetic pixel data only - no model assets.
 */
#include "ai_core/algo_postprocessor.hpp"
#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/algo_types.hpp"
#include "ai_core/image_view.hpp"
#include "gtest/gtest.h"

#include <numeric>
#include <vector>

namespace testing_preproc_cpu {
using namespace ai_core;
using namespace ai_core::dnn;

// RGB image, pixel (y, x, c) = y * 30 + x * 6 + c, packed HWC. Coefficients
// are chosen so every value stays within uint8_t for the sizes used here.
std::vector<uint8_t> makeSyntheticImage(int width, int height, int channels) {
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        pixels[(static_cast<size_t>(y) * width + x) * channels + c] =
            static_cast<uint8_t>(y * 30 + x * 6 + c);
      }
    }
  }
  return pixels;
}

FramePreprocessArg baseArg(int w, int h) {
  FramePreprocessArg arg;
  arg.model_input_shape = {w, h, 3};
  arg.need_resize = false;
  arg.is_equal_scale = false;
  arg.hwc2chw = true;
  arg.data_type = DataType::FLOAT32;
  arg.output_location = BufferLocation::CPU;
  arg.input_names = {"input"};
  return arg;
}

AlgoInput makeInput(const std::vector<uint8_t> &pixels, int w, int h,
                    std::optional<Rect> roi = std::nullopt) {
  ImageView view;
  view.data = pixels.data();
  view.width = w;
  view.height = h;
  view.format = ImagePixelFormat::RGB888;
  FrameInput frame;
  frame.image = view;
  frame.roi = roi;
  AlgoInput input;
  input.setParams(frame);
  return input;
}

TEST(CpuPreprocTest, ChwLayoutAndValues) {
  const int w = 4, h = 4;
  auto pixels = makeSyntheticImage(w, h, 3);

  AlgoPreprocParams params;
  params.setParams(baseArg(w, h));

  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(params), InferErrorCode::SUCCESS);

  auto input = makeInput(pixels, w, h);
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  ASSERT_EQ(preproc.process(input, model_input, ctx), InferErrorCode::SUCCESS);

  ASSERT_TRUE(model_input.contains("input"));
  const Tensor &tensor = model_input.at("input");
  EXPECT_EQ(tensor.shape, (std::vector<int>{1, 3, h, w}));
  ASSERT_EQ(tensor.buffer.getElementCount(), static_cast<size_t>(3 * h * w));

  const float *data = tensor.buffer.getHostPtr<float>();
  // CHW: value at plane c, row y, col x must equal pixel (y, x, c)
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        EXPECT_FLOAT_EQ(data[c * h * w + y * w + x],
                        static_cast<float>(y * 30 + x * 6 + c))
            << "c=" << c << " y=" << y << " x=" << x;
      }
    }
  }

  // The preprocessor publishes the transform context for the postprocessor.
  ASSERT_TRUE(ctx->frame_transform.has_value());
  EXPECT_EQ(ctx->frame_transform->origin_shape.w, w);
  EXPECT_EQ(ctx->frame_transform->roi, (Rect{0, 0, w, h}));
}

TEST(CpuPreprocTest, HwcLayoutKeepsInterleaving) {
  const int w = 2, h = 2;
  auto pixels = makeSyntheticImage(w, h, 3);

  auto arg = baseArg(w, h);
  arg.hwc2chw = false;
  AlgoPreprocParams params;
  params.setParams(arg);

  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(params), InferErrorCode::SUCCESS);

  auto input = makeInput(pixels, w, h);
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  ASSERT_EQ(preproc.process(input, model_input, ctx), InferErrorCode::SUCCESS);

  const Tensor &tensor = model_input.at("input");
  EXPECT_EQ(tensor.shape, (std::vector<int>{1, h, w, 3}));
  const float *data = tensor.buffer.getHostPtr<float>();
  for (size_t i = 0; i < pixels.size(); ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(pixels[i]));
  }
}

TEST(CpuPreprocTest, MeanNormApplied) {
  const int w = 2, h = 2;
  auto pixels = makeSyntheticImage(w, h, 3);

  auto arg = baseArg(w, h);
  arg.mean_vals = {10.f, 10.f, 10.f};
  arg.norm_vals = {2.f, 2.f, 2.f};
  AlgoPreprocParams params;
  params.setParams(arg);

  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(params), InferErrorCode::SUCCESS);

  auto input = makeInput(pixels, w, h);
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  ASSERT_EQ(preproc.process(input, model_input, ctx), InferErrorCode::SUCCESS);

  const float *data = model_input.at("input").buffer.getHostPtr<float>();
  // Plane 0 (c = 0), pixel (0, 0): raw 0 -> (0 - 10) / 2 = -5
  EXPECT_FLOAT_EQ(data[0], -5.f);
  // Plane 0, pixel (0, 1): raw 6 -> (6 - 10) / 2 = -2
  EXPECT_FLOAT_EQ(data[1], -2.f);
}

TEST(CpuPreprocTest, RoiCrop) {
  const int w = 4, h = 4;
  auto pixels = makeSyntheticImage(w, h, 3);

  auto arg = baseArg(2, 2);
  AlgoPreprocParams params;
  params.setParams(arg);

  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(params), InferErrorCode::SUCCESS);

  auto input = makeInput(pixels, w, h, Rect{1, 1, 2, 2});
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  ASSERT_EQ(preproc.process(input, model_input, ctx), InferErrorCode::SUCCESS);

  const float *data = model_input.at("input").buffer.getHostPtr<float>();
  // Cropped (0,0) is source pixel (1,1): value 30 + 6 + 0 = 36
  EXPECT_FLOAT_EQ(data[0], 36.f);
  ASSERT_TRUE(ctx->frame_transform.has_value());
  EXPECT_EQ(ctx->frame_transform->roi, (Rect{1, 1, 2, 2}));
}

TEST(CpuPreprocTest, InvalidRoiFailsGracefully) {
  const int w = 2, h = 2;
  auto pixels = makeSyntheticImage(w, h, 3);

  AlgoPreprocParams params;
  params.setParams(baseArg(w, h));
  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(params), InferErrorCode::SUCCESS);

  // ROI outside the image: plugin throws internally, facade converts to code
  auto input = makeInput(pixels, w, h, Rect{1, 1, 5, 5});
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  EXPECT_EQ(preproc.process(input, model_input, ctx),
            InferErrorCode::InferPreprocessFailed);
}

TEST(CpuPreprocTest, Fp16OutputHasHalfSizedElements) {
  const int w = 2, h = 2;
  auto pixels = makeSyntheticImage(w, h, 3);

  auto arg = baseArg(w, h);
  arg.data_type = DataType::FLOAT16;
  AlgoPreprocParams params;
  params.setParams(arg);

  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(params), InferErrorCode::SUCCESS);

  auto input = makeInput(pixels, w, h);
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  ASSERT_EQ(preproc.process(input, model_input, ctx), InferErrorCode::SUCCESS);

  const auto &buffer = model_input.at("input").buffer;
  EXPECT_EQ(buffer.dataType(), DataType::FLOAT16);
  EXPECT_EQ(buffer.getElementCount(), static_cast<size_t>(3 * h * w));
  EXPECT_EQ(buffer.getSizeBytes(), static_cast<size_t>(3 * h * w) * 2);
}

TEST(CpuPreprocTest, BatchProcessConcatenatesFrames) {
  const int w = 2, h = 2;
  auto pixels_a = makeSyntheticImage(w, h, 3);
  std::vector<uint8_t> pixels_b(pixels_a.size(), 7);

  AlgoPreprocParams params;
  params.setParams(baseArg(w, h));
  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(params), InferErrorCode::SUCCESS);

  std::vector<AlgoInput> inputs{makeInput(pixels_a, w, h),
                                makeInput(pixels_b, w, h)};
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  ASSERT_EQ(preproc.batchProcess(inputs, model_input, ctx),
            InferErrorCode::SUCCESS);

  const Tensor &tensor = model_input.at("input");
  EXPECT_EQ(tensor.shape, (std::vector<int>{2, 3, h, w}));
  const float *data = tensor.buffer.getHostPtr<float>();
  const size_t frame_elems = 3 * h * w;
  EXPECT_FLOAT_EQ(data[0], 0.f);              // frame A pixel (0,0,0)
  EXPECT_FLOAT_EQ(data[frame_elems], 7.f);    // frame B constant fill
  EXPECT_EQ(ctx->frame_transform_batch.size(), 2u);
}

// ============================================================================
// Parameter binding contract (param_validation at initialize)
// ============================================================================

TEST(ParamBindingTest, MonostatePreprocParamsRejected) {
  AlgoPreproc preproc("CpuGenericPreprocess");
  AlgoPreprocParams empty;
  EXPECT_EQ(preproc.initialize(empty), InferErrorCode::InferInvalidInput);
}

TEST(ParamBindingTest, StructurallyInvalidPreprocParamsRejected) {
  AlgoPreproc preproc("CpuGenericPreprocess");

  auto arg = baseArg(4, 4);
  arg.input_names = {"a", "b"}; // exactly one required
  AlgoPreprocParams params;
  params.setParams(arg);
  EXPECT_EQ(preproc.initialize(params), InferErrorCode::InferInvalidInput);

  arg = baseArg(0, 4); // non-positive shape
  AlgoPreprocParams params2;
  params2.setParams(arg);
  EXPECT_EQ(preproc.initialize(params2), InferErrorCode::InferInvalidInput);

  arg = baseArg(4, 4);
  arg.mean_vals = {1.f};
  arg.norm_vals = {1.f, 2.f}; // size mismatch
  AlgoPreprocParams params3;
  params3.setParams(arg);
  EXPECT_EQ(preproc.initialize(params3), InferErrorCode::InferInvalidInput);
}

TEST(ParamBindingTest, MonostatePostprocParamsRejected) {
  AlgoPostprocParams empty;
  ai_core::dnn::AlgoPostproc postproc("Yolov11Det");
  EXPECT_EQ(postproc.initialize(empty), InferErrorCode::InferInvalidInput);
}

TEST(ParamBindingTest, PerCallOverrideWins) {
  const int w = 2, h = 2;
  auto pixels = makeSyntheticImage(w, h, 3);

  // Bind without normalization, override with normalization
  AlgoPreprocParams bound;
  bound.setParams(baseArg(w, h));
  AlgoPreproc preproc("CpuGenericPreprocess");
  ASSERT_EQ(preproc.initialize(bound), InferErrorCode::SUCCESS);

  auto arg = baseArg(w, h);
  arg.mean_vals = {10.f, 10.f, 10.f};
  arg.norm_vals = {2.f, 2.f, 2.f};
  AlgoPreprocParams override_params;
  override_params.setParams(arg);

  auto input = makeInput(pixels, w, h);
  auto ctx = std::make_shared<RuntimeContext>();
  TensorData model_input;
  ASSERT_EQ(preproc.process(input, model_input, ctx, &override_params),
            InferErrorCode::SUCCESS);
  EXPECT_FLOAT_EQ(model_input.at("input").buffer.getHostPtr<float>()[0], -5.f);

  // Without override the bound (no-normalization) params apply
  TensorData model_input2;
  ASSERT_EQ(preproc.process(input, model_input2, ctx), InferErrorCode::SUCCESS);
  EXPECT_FLOAT_EQ(model_input2.at("input").buffer.getHostPtr<float>()[0], 0.f);
}

} // namespace testing_preproc_cpu
