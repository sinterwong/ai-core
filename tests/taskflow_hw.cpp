#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/algo_postproc.hpp"
#include "ai_core/algo_preproc.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <taskflow/taskflow.hpp>
#include <vector>

namespace fs = std::filesystem;

namespace testing_taskflow {

using namespace ai_core;
using namespace ai_core::dnn;

class TaskflowTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {}

  static void TearDownTestSuite() {}
};

TEST_F(TaskflowTest, YoloAlgoProcess) {
  // Create executor and taskflow
  tf::Executor executor;
  tf::Taskflow taskflow("YoloAlgoProcessGraph");

  // Setup algo processor and params (you could load them from configure file)
  AlgoInferParams inferParams;
  std::string inferModuleName;
#ifdef WITH_ORT
  inferModuleName = "OrtAlgoInference";
  inferParams.dataType = DataType::FLOAT16;
  inferParams.modelPath =
      (fs::path("assets") / "models" / "yolov11n-fp16.onnx").string();
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  inferParams.needDecrypt = false;
#elif WITH_NCNN
  inferModuleName = "NCNNAlgoInference";
  inferParams.dataType = DataType::FLOAT32;
  inferParams.modelPath =
      (fs::path("assets") / "models" / "yolov11n.ncnn").string();
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  inferParams.needDecrypt = false;
#elif WITH_TRT
  inferModuleName = "TrtAlgoInference";
  inferParams.dataType = DataType::FLOAT32;
  inferParams.modelPath =
      (fs::path("assets") / "models" / "yolov11n_trt_fp16.engine").string();
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::GPU;
  inferParams.needDecrypt = false;
#else
  GTEST_SKIP() << "No inference backend enabled. Skipping test.";
#endif

  auto engine = std::make_shared<AlgoInferEngine>(inferModuleName, inferParams);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto preproc = std::make_shared<AlgoPreproc>("FramePreprocess");
  ASSERT_EQ(preproc->initialize(), InferErrorCode::SUCCESS);

  auto postproc = std::make_shared<AlgoPostproc>("AnchorDetPostproc");
  ASSERT_EQ(postproc->initialize(), InferErrorCode::SUCCESS);

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = inferParams.dataType;
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = {"images"};
  framePreprocessArg.preprocTaskType =
      FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC;
  framePreprocessArg.outputLocation = BufferLocation::CPU;
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.detAlogType =
      AnchorDetParams::AnchorDetAlogType::YOLO_DET_V11;
  anchorDetParams.condThre = 0.25f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.outputNames = {"output0"};
  postprocParams.setParams(anchorDetParams);

  // Define tasks
  AlgoInput algoInput;
  TensorData modelInput;
  TensorData modelOutput;
  AlgoOutput algoOutput;

  // load image once
  std::string imagePath =
      (fs::path("assets") / "data" / "yolov11/image.png").string();
  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());

  // to RGB -> preproc -> infer -> postproc -> visuzlize
  auto read_and_convert_task =
      taskflow
          .emplace([&]() {
            std::cout << "--- Stage: Read and Convert Image ---\n";
            std::shared_ptr<cv::Mat> image_rgb_ptr =
                std::make_shared<cv::Mat>();
            cv::cvtColor(image, *image_rgb_ptr, cv::COLOR_BGR2RGB);

            FrameInput frameInput;
            frameInput.image = image_rgb_ptr;
            frameInput.inputRoi = std::make_shared<cv::Rect>(
                0, 0, image_rgb_ptr->cols, image_rgb_ptr->rows);
            algoInput.setParams(frameInput);
          })
          .name("Read_Convert_Image");

  auto preprocess_task =
      taskflow
          .emplace([&]() {
            std::cout << "--- Stage: Preprocessing ---\n";
            ASSERT_EQ(preproc->process(algoInput, preprocParams, modelInput),
                      InferErrorCode::SUCCESS);
          })
          .name("Preprocess");

  auto inference_task = taskflow
                            .emplace([&]() {
                              std::cout << "--- Stage: Inference ---\n";
                              ASSERT_EQ(engine->infer(modelInput, modelOutput),
                                        InferErrorCode::SUCCESS);
                            })
                            .name("Inference");

  auto postprocess_task =
      taskflow
          .emplace([&]() {
            std::cout << "--- Stage: Postprocessing ---\n";
            ASSERT_EQ(postproc->process(modelOutput, preprocParams, algoOutput,
                                        postprocParams),
                      InferErrorCode::SUCCESS);
          })
          .name("Postprocess");

  auto visualize_task =
      taskflow
          .emplace([&]() {
            std::cout << "--- Stage: Visualization ---\n";
            auto *detRet = algoOutput.getParams<DetRet>();
            ASSERT_NE(detRet, nullptr);
            ASSERT_EQ(detRet->bboxes.size(), 2);

            cv::Mat visImage = image.clone();
            for (const auto &bbox : detRet->bboxes) {
              cv::rectangle(visImage, *bbox.rect, cv::Scalar(0, 255, 0), 2);
              std::stringstream ss;
              ss << bbox.label << ":" << std::fixed << std::setprecision(2)
                 << bbox.score;
              cv::putText(visImage, ss.str(), bbox.rect->tl(),
                          cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255),
                          2);
            }
            cv::imwrite("taskflow_yolodet_result.png", visImage);
            std::cout << "--- Visualization: Result saved to "
                         "taskflow_yolodet_result.png ---\n";
          })
          .name("Visualize");

  // Define dependencies
  preprocess_task.succeed(read_and_convert_task);
  inference_task.succeed(preprocess_task);
  postprocess_task.succeed(inference_task);
  visualize_task.succeed(postprocess_task);

  // Run the taskflow
  executor.run(taskflow).wait();

  // Clean up (optional, as smart pointers handle most of it)
  engine->terminate();
  preproc->terminate();
  postproc->terminate();
}
} // namespace testing_taskflow
