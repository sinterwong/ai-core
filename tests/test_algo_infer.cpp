#include "ai_core/algo_inference.hpp"
#include "ai_core/opencv_interop.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/input_types.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace testing_algo_infer {
namespace fs = std::filesystem;
using namespace ai_core;
using namespace ai_core::dnn;

void checkResults(const DetRet *det_ret) {
  EXPECT_NE(det_ret, nullptr);
  ASSERT_EQ(det_ret->bboxes.size(), 2);

  const auto &box0 =
      (det_ret->bboxes[0].label == 0) ? det_ret->bboxes[0] : det_ret->bboxes[1];
  const auto &box7 =
      (det_ret->bboxes[0].label == 7) ? det_ret->bboxes[0] : det_ret->bboxes[1];

  EXPECT_EQ(box7.label, 7);
  EXPECT_NEAR(box7.score, 0.54, 1e-2);

  EXPECT_EQ(box0.label, 0);
  EXPECT_NEAR(box0.score, 0.8, 1e-2);
}

TEST(AlgoInferenceTest, YoloDet) {
  fs::path resource_dir = fs::path("assets");
  fs::path data_dir = resource_dir / "data";
  std::string image_path = (data_dir / "yolov11/image.png").string();

  cv::Mat image = cv::imread(image_path);
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoModuleTypes module_types;
  module_types.preproc_module = "CpuGenericPreprocess";
  module_types.postproc_module = "Yolov11Det";

  AlgoInferParams infer_params;
#ifdef WITH_ORT
  module_types.infer_module = "OrtAlgoInference";
  infer_params.data_type = DataType::FLOAT16;
  infer_params.model_path = "assets/models/yolov11n-fp16.onnx";
  infer_params.name = "yolov11n";
  infer_params.device_type = DeviceType::CPU;
  infer_params.need_decrypt = false;
#elif WITH_NCNN
  moduleTypes.inferModule = "NCNNAlgoInference";
  inferParams.dataType = DataType::FLOAT32;
  inferParams.modelPath = "assets/models/yolov11n.ncnn";
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  inferParams.needDecrypt = false;
#elif WITH_TRT
  moduleTypes.inferModule = "TrtAlgoInference";
  inferParams.dataType = DataType::FLOAT32;
  inferParams.modelPath = "assets/models/yolov11n_trt_fp16.engine";
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::GPU;
  inferParams.needDecrypt = false;
#else
  GTEST_SKIP() << "No inference backend enabled. Skipping test.";
#endif

  AlgoInference algo_inf(module_types, infer_params);
  ASSERT_EQ(algo_inf.initialize(), InferErrorCode::SUCCESS);

  FramePreprocessArg frame_preprocess_arg;

  AnchorDetParams anchor_det_params;

#ifdef WITH_ORT
  frame_preprocess_arg.data_type = DataType::FLOAT16;
  frame_preprocess_arg.input_names = {"images"};
  anchor_det_params.output_names = {"output0"};
#elif WITH_NCNN
  frame_preprocess_arg_ptr.dataType = DataType::FLOAT32;
  frame_preprocess_arg_ptr.input_names = {"in0"};
  anchorDetParams.outputNames = {"output0"};
#elif WITH_TRT
  frame_preprocess_arg_ptr.dataType = DataType::FLOAT32;
  frame_preprocess_arg_ptr.input_names = {"images"};
  anchorDetParams.outputNames = {"output0"};
#endif
  frame_preprocess_arg.model_input_shape = {640, 640, 3};
  frame_preprocess_arg.need_resize = true;
  frame_preprocess_arg.is_equal_scale = true;
  frame_preprocess_arg.pad = {0, 0, 0};
  frame_preprocess_arg.mean_vals = {0, 0, 0};
  frame_preprocess_arg.norm_vals = {255.f, 255.f, 255.f};
  frame_preprocess_arg.hwc2chw = true;
  frame_preprocess_arg.output_location = BufferLocation::CPU;

  AlgoPreprocParams preproc_params;
  preproc_params.setParams(frame_preprocess_arg);
  anchor_det_params.cond_thre = 0.5f;
  anchor_det_params.nms_thre = 0.45f;

  AlgoPostprocParams postproc_params;
  postproc_params.setParams(anchor_det_params);

  AlgoInput algo_input;
  FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_rgb);
  frame_input.roi = ai_core::Rect{0, 0, image_rgb.cols, image_rgb.rows};
  algo_input.setParams(frame_input);

  AlgoOutput algo_output;
  ASSERT_EQ(
      algo_inf.infer(algo_input, preproc_params, postproc_params, algo_output),
      InferErrorCode::SUCCESS);

  auto *det_ret = algo_output.getParams<DetRet>();
  checkResults(det_ret);
}

} // namespace testing_algo_infer
