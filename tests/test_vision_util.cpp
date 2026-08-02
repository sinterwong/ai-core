/**
 * @file test_vision_util.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Unit tests for escaleResizeWithPad and box coordinate restoration
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "ai_core/preprocess_types.hpp"
#include "vision_util.hpp"
#include "gtest/gtest.h"
#include <opencv2/core.hpp>

namespace testing_vision_util {

using namespace ai_core;
using ai_core::utils::escaleResizeWithPad;

// Build the context a frame preprocessor would have produced for the given
// letterbox, so the tests exercise exactly what the decoders consume.
FrameTransformContext makeContext(const Shape &origin, const Shape &input,
                                  bool equal_scale, int top_pad = 0,
                                  int left_pad = 0, Rect roi = {}) {
  FrameTransformContext ctx;
  ctx.is_equal_scale = equal_scale;
  ctx.origin_shape = origin;
  ctx.model_input_shape = input;
  ctx.roi = roi;
  ctx.top_pad = top_pad;
  ctx.left_pad = left_pad;
  return ctx;
}

// escaleResizeWithPad must keep aspect ratio, center the resized image and
// report the top/left padding used — postprocessors rely on these offsets to
// map detections back to source coordinates.

TEST(EscaleResizeWithPadTest, LandscapeToSquare) {
  // 200x100 -> 640x640: scale = 3.2, resized 640x320, vertical padding
  cv::Mat src(100, 200, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::Mat dst;
  Shape pad_ret = escaleResizeWithPad(src, dst, 640, 640, {0, 0, 0});

  EXPECT_EQ(dst.cols, 640);
  EXPECT_EQ(dst.rows, 640);
  EXPECT_EQ(pad_ret.w, 0);
  EXPECT_EQ(pad_ret.h, (640 - 320) / 2);
}

TEST(EscaleResizeWithPadTest, PortraitToSquare) {
  // 100x200 -> 640x640: scale = 3.2, resized 320x640, horizontal padding
  cv::Mat src(200, 100, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::Mat dst;
  Shape pad_ret = escaleResizeWithPad(src, dst, 640, 640, {0, 0, 0});

  EXPECT_EQ(dst.cols, 640);
  EXPECT_EQ(dst.rows, 640);
  EXPECT_EQ(pad_ret.w, (640 - 320) / 2);
  EXPECT_EQ(pad_ret.h, 0);
}

TEST(EscaleResizeWithPadTest, OddPaddingSplitsAsymmetrically) {
  // 300x100 -> 640x640: scale = 640/300, resized 640x213 -> 427 rows of
  // padding split 213 top / 214 bottom
  cv::Mat src(100, 300, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat dst;
  Shape pad_ret = escaleResizeWithPad(src, dst, 640, 640, {0, 0, 0});

  EXPECT_EQ(dst.cols, 640);
  EXPECT_EQ(dst.rows, 640);
  const int resized_h = static_cast<int>(100 * (640.f / 300.f));
  EXPECT_EQ(pad_ret.h, (640 - resized_h) / 2);
}

TEST(EscaleResizeWithPadTest, PadValueFillsBorder) {
  cv::Mat src(100, 200, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat dst;
  escaleResizeWithPad(src, dst, 640, 640, {7, 8, 9});

  // Top-left corner lies in the padded band
  const auto &px = dst.at<cv::Vec3b>(0, 0);
  EXPECT_EQ(px[0], 7);
  EXPECT_EQ(px[1], 8);
  EXPECT_EQ(px[2], 9);

  // Center pixel lies in the image band
  const auto &center = dst.at<cv::Vec3b>(320, 320);
  EXPECT_EQ(center[0], 255);
}

TEST(EscaleResizeWithPadTest, NoUpscalePaddingWhenExactFit) {
  cv::Mat src(320, 320, CV_8UC3, cv::Scalar(1, 2, 3));
  cv::Mat dst;
  Shape pad_ret = escaleResizeWithPad(src, dst, 640, 640, {0, 0, 0});

  EXPECT_EQ(pad_ret.w, 0);
  EXPECT_EQ(pad_ret.h, 0);
  EXPECT_EQ(dst.cols, 640);
  EXPECT_EQ(dst.rows, 640);
}

// Round-trip: a point in the source image, mapped through the letterbox
// transform (scale + pad) and restored with FrameTransformContext::mapToSource,
// must land back on the original coordinates. This is exactly what the
// detection postprocessors do.
TEST(CoordinateRestorationTest, RoundTripThroughLetterbox) {
  const Shape origin_shape{200, 100, 3};
  const Shape input_shape{640, 640, 3};

  cv::Mat src(origin_shape.h, origin_shape.w, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat dst;
  Shape pad_ret =
      escaleResizeWithPad(src, dst, input_shape.w, input_shape.h, {0, 0, 0});

  const FrameTransformContext ctx = makeContext(
      origin_shape, input_shape, /*equal_scale=*/true, pad_ret.h, pad_ret.w);

  auto [scaleX, scaleY] = ctx.scaleRatio();
  EXPECT_FLOAT_EQ(scaleX, scaleY); // equal-scale mode

  const float orig_x = 50.f;
  const float orig_y = 25.f;

  // Forward: source -> model input space
  const Point2f model{orig_x * scaleX + pad_ret.w, orig_y * scaleY + pad_ret.h};

  // Backward: model input space -> source
  const Point2f restored = ctx.mapToSource(model);

  EXPECT_NEAR(restored.x, orig_x, 1e-4);
  EXPECT_NEAR(restored.y, orig_y, 1e-4);
}

// With a ROI set, the scale is derived from the ROI (what the model actually
// saw) and the ROI origin is added back on the way out.
TEST(CoordinateRestorationTest, RoundTripThroughRoiAndLetterbox) {
  const Shape origin_shape{1920, 1080, 3};
  const Shape input_shape{640, 640, 3};
  const Rect roi{300, 200, 400, 200};

  const FrameTransformContext ctx =
      makeContext(origin_shape, input_shape, /*equal_scale=*/true,
                  /*top_pad=*/160, /*left_pad=*/0, roi);

  auto [scaleX, scaleY] = ctx.scaleRatio();
  EXPECT_FLOAT_EQ(scaleX, std::min(640.f / 400.f, 640.f / 200.f));
  EXPECT_FLOAT_EQ(scaleX, scaleY);

  // A point at the ROI's top-left maps to the padded model origin and back.
  const Point2f model{0.f + ctx.left_pad, 0.f + ctx.top_pad};
  const Point2f restored = ctx.mapToSource(model);
  EXPECT_NEAR(restored.x, roi.x, 1e-4);
  EXPECT_NEAR(restored.y, roi.y, 1e-4);
}

// Sizes are translation-invariant: no padding subtracted, no ROI offset added.
TEST(CoordinateRestorationTest, SizesIgnorePadAndRoiOffset) {
  const FrameTransformContext ctx =
      makeContext(Shape{200, 100, 3}, Shape{640, 640, 3},
                  /*equal_scale=*/true, /*top_pad=*/160, /*left_pad=*/0,
                  Rect{10, 20, 200, 100});
  auto [scaleX, scaleY] = ctx.scaleRatio();
  const Point2f size = ctx.mapSizeToSource(64.f, 32.f);
  EXPECT_FLOAT_EQ(size.x, 64.f / scaleX);
  EXPECT_FLOAT_EQ(size.y, 32.f / scaleY);
}

TEST(ScaleRatioTest, NonEqualScaleStretchesBothAxes) {
  const FrameTransformContext ctx = makeContext(
      Shape{200, 100, 3}, Shape{640, 640, 3}, /*equal_scale=*/false);
  auto [scaleX, scaleY] = ctx.scaleRatio();
  EXPECT_FLOAT_EQ(scaleX, 640.f / 200.f);
  EXPECT_FLOAT_EQ(scaleY, 640.f / 100.f);
}

TEST(ScaleRatioTest, EqualScaleUsesMinRatioOnBothAxes) {
  const FrameTransformContext ctx =
      makeContext(Shape{200, 100, 3}, Shape{640, 640, 3}, /*equal_scale=*/true);
  auto [rw, rh] = ctx.scaleRatio();
  EXPECT_FLOAT_EQ(rw, rh);
  EXPECT_FLOAT_EQ(rw, std::min(640.f / 200.f, 640.f / 100.f));
}

// A degenerate context must not divide by zero — it degrades to identity.
TEST(ScaleRatioTest, EmptySourceShapeDegradesToIdentity) {
  const FrameTransformContext ctx =
      makeContext(Shape{0, 0, 3}, Shape{640, 640, 3}, /*equal_scale=*/true);
  auto [rw, rh] = ctx.scaleRatio();
  EXPECT_FLOAT_EQ(rw, 1.f);
  EXPECT_FLOAT_EQ(rh, 1.f);
}

// ============================================================================
// NMS + IoU
// ============================================================================

TEST(NmsTest, CalculateIoUOverlap) {
  BBox a{Rect{0, 0, 10, 10}, 0.9f, 0};
  BBox b{Rect{5, 0, 10, 10}, 0.8f, 0};
  // Intersection 5x10 = 50, union = 100 + 100 - 50 = 150
  EXPECT_NEAR(ai_core::utils::calculateIoU(a, b), 50.f / 150.f, 1e-5);

  BBox c{Rect{100, 100, 10, 10}, 0.7f, 0};
  EXPECT_FLOAT_EQ(ai_core::utils::calculateIoU(a, c), 0.f); // disjoint
}

TEST(NmsTest, SuppressesOverlappingSameClass) {
  std::vector<BBox> boxes{
      {Rect{0, 0, 10, 10}, 0.9f, 0},
      {Rect{1, 1, 10, 10}, 0.8f, 0}, // heavy overlap with the first, class 0
      {Rect{50, 50, 10, 10}, 0.7f, 0},
  };
  auto kept =
      ai_core::utils::nms(boxes, /*nms_thre=*/0.45f, /*conf_thre=*/0.5f);
  // The overlapping lower-score box is suppressed; the distant one survives.
  EXPECT_EQ(kept.size(), 2u);
}

TEST(NmsTest, KeepsDifferentClassesEvenIfOverlapping) {
  std::vector<BBox> boxes{
      {Rect{0, 0, 10, 10}, 0.9f, 0},
      {Rect{0, 0, 10, 10}, 0.8f, 1}, // same box, different class
  };
  auto kept = ai_core::utils::nms(boxes, 0.45f, 0.5f);
  EXPECT_EQ(kept.size(), 2u);
}

} // namespace testing_vision_util
