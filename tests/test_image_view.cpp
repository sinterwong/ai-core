/**
 * @file test_image_view.cpp
 * @brief Unit tests for ImageView and the OpenCV interop layer.
 * No model assets required.
 */
#include "ai_core/image_view.hpp"
#include "ai_core/opencv_interop.hpp"
#include "gtest/gtest.h"

#include <opencv2/core.hpp>

namespace testing_image_view {
using namespace ai_core;

TEST(ImageViewTest, ChannelCountPerFormat) {
  EXPECT_EQ(channelCount(ImagePixelFormat::GRAY8), 1);
  EXPECT_EQ(channelCount(ImagePixelFormat::BGR888), 3);
  EXPECT_EQ(channelCount(ImagePixelFormat::RGB888), 3);
  EXPECT_EQ(channelCount(ImagePixelFormat::BGRA8888), 4);
  EXPECT_EQ(channelCount(ImagePixelFormat::RGBA8888), 4);
}

TEST(ImageViewTest, StrideZeroMeansPacked) {
  ImageView view;
  view.width = 10;
  view.height = 4;
  view.format = ImagePixelFormat::BGR888;
  view.stride = 0;
  EXPECT_EQ(view.strideBytes(), 30u);

  view.stride = 64;
  EXPECT_EQ(view.strideBytes(), 64u);
}

TEST(ImageViewTest, EmptySemantics) {
  ImageView view;
  EXPECT_TRUE(view.empty());

  uint8_t px = 0;
  view.data = &px;
  view.width = 1;
  view.height = 1;
  EXPECT_FALSE(view.empty());

  view.height = 0;
  EXPECT_TRUE(view.empty());
}

TEST(InteropTest, ViewFromMatIsZeroCopy) {
  cv::Mat mat(4, 6, CV_8UC3, cv::Scalar(1, 2, 3));
  ImageView view = interop::viewFromMat(mat);

  EXPECT_EQ(view.data, mat.data);
  EXPECT_EQ(view.width, 6);
  EXPECT_EQ(view.height, 4);
  EXPECT_EQ(view.strideBytes(), mat.step);
  EXPECT_EQ(view.format, ImagePixelFormat::BGR888);
}

TEST(InteropTest, ViewFromMatFormatDefaults) {
  cv::Mat gray(2, 2, CV_8UC1);
  EXPECT_EQ(interop::viewFromMat(gray).format, ImagePixelFormat::GRAY8);

  cv::Mat bgra(2, 2, CV_8UC4);
  EXPECT_EQ(interop::viewFromMat(bgra).format, ImagePixelFormat::BGRA8888);
}

TEST(InteropTest, ViewFromMatExplicitFormat) {
  cv::Mat mat(2, 2, CV_8UC3);
  ImageView view = interop::viewFromMat(mat, ImagePixelFormat::RGB888);
  EXPECT_EQ(view.format, ImagePixelFormat::RGB888);

  // Channel count must match the requested format
  EXPECT_THROW(interop::viewFromMat(mat, ImagePixelFormat::GRAY8),
               std::invalid_argument);
}

TEST(InteropTest, ViewFromMatRejectsNon8Bit) {
  cv::Mat floats(2, 2, CV_32FC3);
  EXPECT_THROW(interop::viewFromMat(floats), std::invalid_argument);

  cv::Mat two_channels(2, 2, CV_8UC2);
  EXPECT_THROW(interop::viewFromMat(two_channels), std::invalid_argument);
}

TEST(InteropTest, EmptyMatGivesEmptyView) {
  cv::Mat empty;
  EXPECT_TRUE(interop::viewFromMat(empty).empty());
  EXPECT_TRUE(interop::matFromView(ImageView{}).empty());
}

TEST(InteropTest, MatFromViewRoundTrip) {
  cv::Mat mat(3, 5, CV_8UC3);
  cv::randu(mat, 0, 255);

  ImageView view = interop::viewFromMat(mat);
  cv::Mat wrapped = interop::matFromView(view);

  EXPECT_EQ(wrapped.data, mat.data); // zero copy
  EXPECT_EQ(wrapped.cols, 5);
  EXPECT_EQ(wrapped.rows, 3);
  EXPECT_EQ(wrapped.channels(), 3);
  EXPECT_EQ(wrapped.step, mat.step);
}

TEST(InteropTest, MatFromViewRespectsStride) {
  // Simulate a strided view: 4x2 RGB rows padded to 16 bytes
  std::vector<uint8_t> pixels(4 * 16, 0);
  pixels[0] = 42; // (0,0) B
  pixels[16] = 7; // (1,0) B

  ImageView view;
  view.data = pixels.data();
  view.width = 2;
  view.height = 4;
  view.stride = 16;
  view.format = ImagePixelFormat::BGR888;

  cv::Mat wrapped = interop::matFromView(view);
  EXPECT_EQ(wrapped.step, 16u);
  EXPECT_EQ(wrapped.at<cv::Vec3b>(0, 0)[0], 42);
  EXPECT_EQ(wrapped.at<cv::Vec3b>(1, 0)[0], 7);
}

TEST(InteropTest, GeometryConversions) {
  Rect r{1, 2, 30, 40};
  cv::Rect cr = interop::toCv(r);
  EXPECT_EQ(cr, cv::Rect(1, 2, 30, 40));
  EXPECT_EQ(interop::fromCv(cr), r);

  Point p{3, 4};
  cv::Point cp = interop::toCv(p);
  EXPECT_EQ(cp, cv::Point(3, 4));
  Point back = interop::fromCv(cp);
  EXPECT_EQ(back.x, 3);
  EXPECT_EQ(back.y, 4);

  Point2f pf{1.5f, 2.5f};
  cv::Point2f cpf = interop::toCv(pf);
  EXPECT_FLOAT_EQ(cpf.x, 1.5f);
  Point2f backf = interop::fromCv(cpf);
  EXPECT_FLOAT_EQ(backf.y, 2.5f);
}

TEST(GeometryTest, RectHelpers) {
  Rect r{0, 0, 4, 5};
  EXPECT_EQ(r.area(), 20);
  EXPECT_FALSE(r.empty());

  Rect zero{};
  EXPECT_TRUE(zero.empty());
  EXPECT_EQ(zero.area(), 0);
}

} // namespace testing_image_view
