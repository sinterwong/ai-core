#include "ai_core/algo_inference.hpp"
#include "ai_core/config/algo_config.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include "ai_core/plugin_manager.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <config.json> <image>\n";
    return 2;
  }
  const std::string config_path = argv[1];
  const std::string image_path = argv[2];

  using namespace ai_core;

  // Maintained plugins are separate DSOs in v2; loading the install directory
  // makes their registered names available before constructing the pipeline.
  dnn::PluginManager::instance().loadDirectory(AI_CORE_PLUGIN_DIR);

  // 1) Load + validate the pipeline definition from JSON (no C++ wiring).
  config::AlgoConfig cfg;
  try {
    cfg = config::loadAlgoConfig(config_path);
  } catch (const std::exception &e) {
    std::cerr << "config error: " << e.what() << "\n";
    return 1;
  }

  // 2) Build the pipeline and bind params once.
  dnn::AlgoInference engine(cfg.module_types, cfg.infer_params);
  if (engine.initialize(cfg.preproc_params, cfg.postproc_params) !=
      InferErrorCode::SUCCESS) {
    std::cerr << "engine init failed\n";
    return 1;
  }

  // 3) Read an image and wrap it as a (non-owning) ImageView.
  cv::Mat bgr = cv::imread(image_path);
  if (bgr.empty()) {
    std::cerr << "failed to read image: " << image_path << "\n";
    return 1;
  }

  AlgoInput input;
  FrameInput frame;
  // The preprocessor converts BGR to whatever the config declares as
  // inputFormat, so imread's output goes in as-is. bgr must outlive the
  // infer call.
  frame.image = interop::viewFromMat(bgr);
  input.setParams(frame);

  // 4) Infer (data only — params are already bound).
  AlgoOutput output;
  if (auto ec = engine.infer(input, output); ec != InferErrorCode::SUCCESS) {
    std::cerr << "infer failed: " << to_string(ec) << "\n";
    return 1;
  }

  // 5) Consume the typed result.
  if (const auto *det = output.getParams<DetRet>()) {
    std::cout << "detections: " << det->bboxes.size() << "\n";
    for (const auto &b : det->bboxes) {
      std::cout << "  label=" << b.label << " score=" << b.score << " rect=["
                << b.rect.x << "," << b.rect.y << "," << b.rect.width << ","
                << b.rect.height << "]\n";
    }
  } else {
    std::cout << "inference ok (non-detection output type)\n";
  }
  return 0;
}
