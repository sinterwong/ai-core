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
#include "vision_util.hpp"
#include "gtest/gtest.h"
#include <opencv2/core.hpp>

namespace testing_vision_util {

using namespace ai_core;
using ai_core::utils::escaleResizeWithPad;
using ai_core::utils::scaleRatio;

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
// transform (scale + pad) and restored with scaleRatio + pad offsets, must
// land back on the original coordinates. This mirrors what the detection
// postprocessors do.
TEST(CoordinateRestorationTest, RoundTripThroughLetterbox) {
  const Shape origin_shape{200, 100, 3};
  const Shape input_shape{640, 640, 3};

  cv::Mat src(origin_shape.h, origin_shape.w, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat dst;
  Shape pad_ret = escaleResizeWithPad(src, dst, input_shape.w, input_shape.h,
                                      {0, 0, 0});

  auto [scaleX, scaleY] = scaleRatio(origin_shape, input_shape, true);
  EXPECT_FLOAT_EQ(scaleX, scaleY); // equal-scale mode

  const float orig_x = 50.f;
  const float orig_y = 25.f;

  // Forward: source -> model input space
  const float model_x = orig_x * scaleX + pad_ret.w;
  const float model_y = orig_y * scaleY + pad_ret.h;

  // Backward: model input space -> source (the postprocessor formula)
  const float restored_x = (model_x - pad_ret.w) / scaleX;
  const float restored_y = (model_y - pad_ret.h) / scaleY;

  EXPECT_NEAR(restored_x, orig_x, 1e-4);
  EXPECT_NEAR(restored_y, orig_y, 1e-4);
}

TEST(ScaleRatioTest, NonEqualScaleStretchesBothAxes) {
  const Shape origin_shape{200, 100, 3};
  const Shape input_shape{640, 640, 3};
  auto [scaleX, scaleY] = scaleRatio(origin_shape, input_shape, false);
  EXPECT_FLOAT_EQ(scaleX, 640.f / 200.f);
  EXPECT_FLOAT_EQ(scaleY, 640.f / 100.f);
}

} // namespace testing_vision_util
