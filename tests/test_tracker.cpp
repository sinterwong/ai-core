/**
 * @file test_tracker.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-03-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "iou_associator.hpp"
#include "track_default.hpp"
#include "gtest/gtest.h"

using namespace infer::tracker;
using namespace infer;

namespace test_tracker {
void AssertIndexVectorsEqual(std::vector<int> vec1, std::vector<int> vec2) {
  std::sort(vec1.begin(), vec1.end());
  std::sort(vec2.begin(), vec2.end());
  ASSERT_EQ(vec1, vec2);
}

void AssertMatchVectorsEqual(std::vector<std::pair<int, int>> vec1,
                             std::vector<std::pair<int, int>> vec2) {
  auto sort_key = [](const std::pair<int, int> &a,
                     const std::pair<int, int> &b) {
    return a.first < b.first;
  };
  std::sort(vec1.begin(), vec1.end(), sort_key);
  std::sort(vec2.begin(), vec2.end(), sort_key);
  ASSERT_EQ(vec1, vec2);
}

class MockDetection : public IDetection {
public:
  MockDetection(BBox bbox, int id, int frameIndex)
      : bbox_(bbox), id_(id), frameIndex_(frameIndex) {}

  BBox getBoundingBox() const override { return bbox_; }

  int getLabel() const override { return bbox_.label; }

  float getScore() const override { return bbox_.score; }

  int getId() const override { return id_; }

  int getFrameIndex() const override { return frameIndex_; }

private:
  BBox bbox_;
  int label_;
  float score_;
  int id_;
  int frameIndex_;
};

class TrackerTest : public ::testing::Test {
protected:
  TrackingConfig config;
  IoUAssociator associator{0.5f};

  // IoU > 0.5
  BBox box_high_iou_1 = BBox{cv::Rect{0, 0, 10, 10}, 0.9f, 0};
  BBox box_high_iou_2 = BBox{cv::Rect{1, 1, 10, 10}, 0.8f, 0};

  // IoU < 0.5
  BBox box_low_iou_1 = BBox{cv::Rect{0, 0, 10, 10}, 0.9f, 0};
  BBox box_low_iou_2 = BBox{cv::Rect{5, 0, 10, 10}, 0.95f, 0};

  // completely separate
  BBox box_no_iou_1 = BBox{cv::Rect{0, 0, 10, 10}, 0.8f, 0};
  BBox box_no_iou_2 = BBox{cv::Rect{20, 20, 5, 5}, 0.8f, 0};

  // high IoU pair but different label
  BBox box_high_iou_3 = BBox{cv::Rect{50, 50, 10, 10}, 0.7f, 0};
  BBox box_high_iou_4 = BBox{cv::Rect{51, 51, 10, 10}, 0.8f, 1};

  void SetUp() override {
    // configure
    config.similarityThreshold = 0.5f;
    config.activationThreshold = 0.7f;
    config.inactivationThreshold = 0.3f;
    config.terminationThreshold = 0.1f;
    config.weakDetectionThreshold = 0.2f;
    config.displayThreshold = 0.3f;
    config.maxHistorySize = 5;
    config.maxConsecutiveMisses = 3;
    config.scoreAverageWindowSize = 3;
  }
  void TearDown() override {}
};

TEST_F(TrackerTest, IoUAssociate_SimpleMatch) {
  // arrange
  auto initialDet = std::make_shared<MockDetection>(box_high_iou_1, 10, 0);
  auto trackable = std::make_shared<DefaultTrackable>(initialDet, 100, config);
  std::vector<std::shared_ptr<ITrackable>> trackables = {trackable};

  auto currentDet = std::make_shared<MockDetection>(box_high_iou_2, 20, 1);
  std::vector<std::shared_ptr<IDetection>> detections = {currentDet};

  // act
  AssociationResult result = associator.associate(trackables, detections);

  // assert
  ASSERT_EQ(result.matches.size(), 1);
  AssertMatchVectorsEqual(result.matches, {{0, 0}});
  ASSERT_EQ(result.unmatchedDetections.size(), 0);
  ASSERT_EQ(result.unmatchedTrackables.size(), 0);
}

TEST_F(TrackerTest, IoUAssociate_NoMatchLowIOU) {
  // arrange
  auto initialDet = std::make_shared<MockDetection>(box_low_iou_1, 10, 0);
  auto trackable = std::make_shared<DefaultTrackable>(initialDet, 100, config);
  std::vector<std::shared_ptr<ITrackable>> trackables = {trackable};

  auto currentDet = std::make_shared<MockDetection>(box_low_iou_2, 20, 1);
  std::vector<std::shared_ptr<IDetection>> detections = {currentDet};

  AssociationResult result = associator.associate(trackables, detections);

  // assert
  ASSERT_EQ(result.matches.size(), 0);
  ASSERT_EQ(result.unmatchedTrackables.size(), 1);
  ASSERT_EQ(result.unmatchedDetections.size(), 1);
  AssertIndexVectorsEqual(result.unmatchedDetections, {0});
}
} // namespace test_tracker
