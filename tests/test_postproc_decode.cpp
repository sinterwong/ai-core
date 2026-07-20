/**
 * @file test_postproc_decode.cpp
 * @brief Unit tests for every postprocessor's decode logic, driven by
 * synthetic tensors through the AlgoPostproc facade. No model assets.
 */
#include "ai_core/algo_postprocessor.hpp"
#include "ai_core/algo_types.hpp"
#include "gtest/gtest.h"

#include <cmath>
#include <cstring>

namespace testing_postproc_decode {
using namespace ai_core;
using namespace ai_core::dnn;

TypedBuffer floatBuffer(const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return TypedBuffer::createFromCpu(DataType::FLOAT32, std::move(bytes));
}

TypedBuffer int64Buffer(const std::vector<int64_t> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return TypedBuffer::createFromCpu(DataType::INT64, std::move(bytes));
}

// Identity transform: model input == source image, no scaling, no padding.
std::shared_ptr<RuntimeContext> identityContext(int w, int h) {
  auto ctx = std::make_shared<RuntimeContext>();
  FrameTransformContext t;
  t.is_equal_scale = false;
  t.origin_shape = {w, h, 3};
  t.model_input_shape = {w, h, 3};
  t.roi = Rect{0, 0, w, h};
  t.top_pad = 0;
  t.left_pad = 0;
  ctx->frame_transform = t;
  ctx->frame_transform_batch = {t};
  return ctx;
}

// ============================================================================
// SoftmaxCls
// ============================================================================

TEST(SoftmaxClsDecode, PicksArgmaxWithSoftmaxScore) {
  TensorData model_output;
  model_output.set("logits", floatBuffer({0.f, 2.f, 1.f}), {1, 3});

  GenericPostParams params;
  params.output_names = {"logits"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("SoftmaxCls");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(4, 4);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *cls = output.getParams<ClsRet>();
  ASSERT_NE(cls, nullptr);
  EXPECT_EQ(cls->label, 1);
  // softmax([0,2,1])[1] = e^2 / (1 + e^2 + e)
  const float expected =
      std::exp(2.f) / (1.f + std::exp(2.f) + std::exp(1.f));
  EXPECT_NEAR(cls->score, expected, 1e-5);
}

TEST(SoftmaxClsDecode, BatchDecodesEachRow) {
  TensorData model_output;
  model_output.set("logits", floatBuffer({5.f, 0.f, /*row1*/ 0.f, 3.f}),
                   {2, 2});

  GenericPostParams params;
  params.output_names = {"logits"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("SoftmaxCls");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(4, 4);
  std::vector<AlgoOutput> outputs;
  ASSERT_EQ(postproc.batchProcess(model_output, outputs, ctx),
            InferErrorCode::SUCCESS);

  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(outputs[0].getParams<ClsRet>()->label, 0);
  EXPECT_EQ(outputs[1].getParams<ClsRet>()->label, 1);
}

// ============================================================================
// FprCls
// ============================================================================

TEST(FprClsDecode, ArgmaxOnScoresAndBirads) {
  TensorData model_output;
  model_output.set("scores", floatBuffer({0.1f, 0.7f, 0.2f}), {1, 3});
  model_output.set("birads", floatBuffer({0.2f, 0.1f, 0.3f, 0.4f}), {1, 4});

  GenericPostParams params;
  params.output_names = {"scores", "birads"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("FprCls");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(4, 4);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *fpr = output.getParams<FprClsRet>();
  ASSERT_NE(fpr, nullptr);
  EXPECT_EQ(fpr->label, 1);
  EXPECT_NEAR(fpr->score, 0.7f, 1e-6);
  EXPECT_EQ(fpr->birad, 3);
  ASSERT_EQ(fpr->score_probs.size(), 3u);
  EXPECT_NEAR(fpr->score_probs[2], 0.2f, 1e-6);
}

// ============================================================================
// OCRReco (CTC collapse)
// ============================================================================

TEST(OcrRecoDecode, CtcRemovesBlanksAndRepeats) {
  TensorData model_output;
  model_output.set("lengths", int64Buffer({5}), {1});
  model_output.set("argmax", int64Buffer({0, 4, 4, 0, 7, 9, 9, 0}), {1, 8});

  GenericPostParams params;
  params.output_names = {"lengths", "argmax"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("OCRReco");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(4, 4);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *ocr = output.getParams<OCRRecoRet>();
  ASSERT_NE(ocr, nullptr);
  EXPECT_EQ(ocr->output_lengths, 5);
  EXPECT_EQ(ocr->outputs, (std::vector<int64_t>{4, 7, 9}));
}

// ============================================================================
// RawModelOutput passthrough
// ============================================================================

TEST(RawOutputDecode, PassesTensorsThrough) {
  TensorData model_output;
  model_output.set("feat", floatBuffer({1.f, 2.f, 3.f}), {1, 3});

  GenericPostParams params; // names unused by RawModelOutput
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("RawModelOutput");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(4, 4);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *raw = output.getParams<RawModelOutput>();
  ASSERT_NE(raw, nullptr);
  ASSERT_TRUE(raw->contains("feat"));
  EXPECT_EQ(raw->at("feat").shape, (std::vector<int>{1, 3}));
  EXPECT_FLOAT_EQ(raw->at("feat").buffer.getHostPtr<float>()[2], 3.f);
}

// ============================================================================
// Yolov11Det
// ============================================================================

TEST(YoloDetDecode, DecodesSingleStrongBox) {
  // Layout [1, 4 + nc, anchors], attribute-major: 2 classes, 3 anchors.
  const int nc = 2, anchors = 3;
  std::vector<float> data(static_cast<size_t>(4 + nc) * anchors, 0.f);
  auto at = [&](int attr, int anchor) -> float & {
    return data[attr * anchors + anchor];
  };
  // Anchor 1: center (320, 300), size 64x32, class 1 score 0.9
  at(0, 1) = 320.f;
  at(1, 1) = 300.f;
  at(2, 1) = 64.f;
  at(3, 1) = 32.f;
  at(4 + 1, 1) = 0.9f;
  // Anchor 0/2 stay below threshold.
  at(4 + 0, 0) = 0.2f;

  TensorData model_output;
  model_output.set("output0", floatBuffer(data), {1, 4 + nc, anchors});

  AnchorDetParams params;
  params.cond_thre = 0.5f;
  params.nms_thre = 0.45f;
  params.output_names = {"output0"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("Yolov11Det");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(640, 640);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *det = output.getParams<DetRet>();
  ASSERT_NE(det, nullptr);
  ASSERT_EQ(det->bboxes.size(), 1u);
  const BBox &box = det->bboxes[0];
  EXPECT_EQ(box.label, 1);
  EXPECT_NEAR(box.score, 0.9f, 1e-6);
  EXPECT_EQ(box.rect, (Rect{320 - 32, 300 - 16, 64, 32}));
}

TEST(YoloDetDecode, EqualScaleMapsBackThroughPad) {
  // Source image 320x640 letterboxed into 640x640: scale = 1, left_pad = 160.
  const int nc = 1, anchors = 1;
  std::vector<float> data(static_cast<size_t>(4 + nc) * anchors, 0.f);
  data[0] = 320.f; // cx in model space
  data[1] = 100.f; // cy
  data[2] = 40.f;  // w
  data[3] = 20.f;  // h
  data[4] = 0.8f;  // class 0 score

  TensorData model_output;
  model_output.set("output0", floatBuffer(data), {1, 4 + nc, anchors});

  AnchorDetParams params;
  params.cond_thre = 0.5f;
  params.nms_thre = 0.45f;
  params.output_names = {"output0"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("Yolov11Det");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = std::make_shared<RuntimeContext>();
  FrameTransformContext t;
  t.is_equal_scale = true;
  t.origin_shape = {320, 640, 3};
  t.model_input_shape = {640, 640, 3};
  t.roi = Rect{0, 0, 320, 640};
  t.left_pad = 160;
  t.top_pad = 0;
  ctx->frame_transform = t;

  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *det = output.getParams<DetRet>();
  ASSERT_EQ(det->bboxes.size(), 1u);
  // x = (cx - w/2 - left_pad) / 1 = 320 - 20 - 160 = 140
  EXPECT_EQ(det->bboxes[0].rect, (Rect{140, 90, 40, 20}));
}

TEST(YoloDetDecode, MissingTransformContextFails) {
  TensorData model_output;
  model_output.set("output0", floatBuffer({0.f}), {1, 1, 1});

  AnchorDetParams params;
  params.cond_thre = 0.5f;
  params.nms_thre = 0.45f;
  params.output_names = {"output0"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("Yolov11Det");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = std::make_shared<RuntimeContext>(); // no frame_transform
  AlgoOutput output;
  EXPECT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::InferInvalidInput);
}

// ============================================================================
// NanoDet
// ============================================================================

TEST(NanoDetDecode, DecodesCornerBoxLayout) {
  // Layout [1, anchors, nc + 4], anchor-major rows: [scores..., x1,y1,x2,y2]
  const int nc = 3, anchors = 2;
  std::vector<float> data(static_cast<size_t>(anchors) * (nc + 4), 0.f);
  // Anchor 0: class 2 score 0.85, box (10, 20) - (110, 70)
  data[2] = 0.85f;
  data[nc + 0] = 10.f;
  data[nc + 1] = 20.f;
  data[nc + 2] = 110.f;
  data[nc + 3] = 70.f;

  TensorData model_output;
  model_output.set("output", floatBuffer(data), {1, anchors, nc + 4});

  AnchorDetParams params;
  params.cond_thre = 0.5f;
  params.nms_thre = 0.45f;
  params.output_names = {"output"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("NanoDet");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(640, 640);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *det = output.getParams<DetRet>();
  ASSERT_NE(det, nullptr);
  ASSERT_EQ(det->bboxes.size(), 1u);
  EXPECT_EQ(det->bboxes[0].label, 2);
  EXPECT_EQ(det->bboxes[0].rect, (Rect{10, 20, 100, 50}));
}

// ============================================================================
// RTMDet
// ============================================================================

TEST(RtmDetDecode, DecodesSplitDetClsTensors) {
  const int nc = 2, anchors = 2;
  // det: [1, anchors, 4] corner boxes
  std::vector<float> det_data{
      50.f, 60.f, 150.f, 120.f, // anchor 0
      0.f,  0.f,  0.f,   0.f,   // anchor 1
  };
  // cls: [1, anchors, nc]
  std::vector<float> cls_data{
      0.1f, 0.75f, // anchor 0 -> class 1
      0.f,  0.f,   // anchor 1 below threshold
  };

  TensorData model_output;
  model_output.set("dets", floatBuffer(det_data), {1, anchors, 4});
  model_output.set("labels", floatBuffer(cls_data), {1, anchors, nc});

  AnchorDetParams params;
  params.cond_thre = 0.5f;
  params.nms_thre = 0.45f;
  params.output_names = {"dets", "labels"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("RTMDet");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(640, 640);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *det = output.getParams<DetRet>();
  ASSERT_NE(det, nullptr);
  ASSERT_EQ(det->bboxes.size(), 1u);
  EXPECT_EQ(det->bboxes[0].label, 1);
  EXPECT_EQ(det->bboxes[0].rect, (Rect{50, 60, 100, 60}));
}

// ============================================================================
// SemanticSeg
// ============================================================================

TEST(SemanticSegDecode, ExtractsContourForConfidentClass) {
  // [1, 2, 4, 4]: class 1 confident in the lower-right 2x2 block.
  const int nc = 2, h = 4, w = 4;
  std::vector<float> data(static_cast<size_t>(nc) * h * w, 0.f);
  // class 0 background: prob 0.9 everywhere
  for (int i = 0; i < h * w; ++i) {
    data[i] = 0.9f;
  }
  // class 1 block
  for (int y = 2; y < 4; ++y) {
    for (int x = 2; x < 4; ++x) {
      data[0 * h * w + y * w + x] = 0.1f;
      data[1 * h * w + y * w + x] = 0.9f;
    }
  }

  TensorData model_output;
  model_output.set("seg", floatBuffer(data), {1, nc, h, w});

  ConfidenceFilterParams params;
  params.cond_thre = 0.5f;
  params.output_names = {"seg"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("SemanticSeg");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(w, h);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *seg = output.getParams<SegRet>();
  ASSERT_NE(seg, nullptr);
  ASSERT_TRUE(seg->cls_to_contours.count(1));
  ASSERT_FALSE(seg->cls_to_contours.at(1).empty());
  // All contour points must fall inside the 2x2 block
  for (const auto &pt : seg->cls_to_contours.at(1).front()) {
    EXPECT_GE(pt.x, 2);
    EXPECT_LE(pt.x, 3);
    EXPECT_GE(pt.y, 2);
    EXPECT_LE(pt.y, 3);
  }
}

// ============================================================================
// UNetDualOutputSeg
// ============================================================================

TEST(UnetDualSegDecode, ReturnsOwningTensors) {
  // Shapes are {1, W, H}; the decoder reads height = shape[2].
  const int w = 2, h = 3;
  std::vector<float> prob(static_cast<size_t>(w) * h);
  std::vector<float> mask(static_cast<size_t>(w) * h);
  for (size_t i = 0; i < prob.size(); ++i) {
    prob[i] = 0.1f * static_cast<float>(i);
    mask[i] = static_cast<float>(i % 2);
  }

  TensorData model_output;
  model_output.set("prob", floatBuffer(prob), {1, w, h});
  model_output.set("mask", floatBuffer(mask), {1, w, h});

  GenericPostParams params;
  params.output_names = {"prob", "mask"};
  AlgoPostprocParams post_params;
  post_params.setParams(params);

  AlgoPostproc postproc("UNetDualOutputSeg");
  ASSERT_EQ(postproc.initialize(post_params), InferErrorCode::SUCCESS);

  auto ctx = identityContext(8, 8);
  AlgoOutput output;
  ASSERT_EQ(postproc.process(model_output, output, ctx),
            InferErrorCode::SUCCESS);

  const auto *dual = output.getParams<DualRawSegRet>();
  ASSERT_NE(dual, nullptr);
  EXPECT_EQ(dual->prob.shape, (std::vector<int>{h, w}));
  EXPECT_EQ(dual->mask.shape, (std::vector<int>{h, w}));

  // The tensors own copies: still valid and equal to the source data
  const float *prob_out = dual->prob.buffer.getHostPtr<float>();
  for (size_t i = 0; i < prob.size(); ++i) {
    EXPECT_FLOAT_EQ(prob_out[i], prob[i]);
  }
}

} // namespace testing_postproc_decode
