#include "ai_core/plugin_manager.hpp"
#include "ai_core/runtime.hpp"
#include "ocr_utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

#ifdef AI_CORE_EXAMPLE_PREPROC_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_EXAMPLE_PREPROC_PLUGIN);
  ai_core::dnn::PluginManager::instance().load(AI_CORE_EXAMPLE_POSTPROC_PLUGIN);
#endif
#ifdef AI_CORE_EXAMPLE_ORT_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_EXAMPLE_ORT_PLUGIN);
#endif
#ifdef AI_CORE_EXAMPLE_NCNN_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_EXAMPLE_NCNN_PLUGIN);
#endif

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <config_path> <image_path> <dict_path>" << std::endl;
    return -1;
  }

  std::string det_config_path = argv[1];
  std::string rec_config_path = argv[2];
  std::string image_path = argv[3];
  std::string dict_path = argv[4];

  ai_core::LoggingConfig logging;
  logging.min_level = ai_core::LogLevel::Trace;
  logging.console_enabled = true;
  logging.file_enabled = false;
  logging.color_enabled = true;
  ai_core::Runtime::configureLogging(logging);

  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    std::cerr << "Failed to read image: " << image_path;
    return -1;
  }

  try {
    ai_core::example::OCRUtils *ocr = ai_core::example::OCRUtils::instance(
        det_config_path, rec_config_path, dict_path);

    auto detected_boxes = ocr->detect(image);
    std::clog << "Detected BBoxes: " << detected_boxes.size();

    std::vector<std::string> recognized_texts;
    for (auto &bbox : detected_boxes) {
      bbox = ocr->expandBox(bbox, 0.0f, 0.5f, image.size());
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        std::cerr << "Expanded bounding box is empty or has zero dimension, "
                     "skipping.";
        continue;
      }
      // make sure the bbox is inside the image
      bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        std::cerr << "Expanded bounding box clipped to image boundaries is "
                     "empty or has zero dimension, skipping.";
        continue;
      }

      cv::Mat textImage = image(bbox);
      cv::Mat grayImage;
      if (textImage.channels() == 3) {
        cv::cvtColor(textImage, grayImage, cv::COLOR_BGR2GRAY);
      } else {
        grayImage = textImage;
      }
      std::vector<std::string> texts = ocr->recognize(grayImage);
      std::string text;
      for (const auto &t : texts) {
        text += "-" + t;
      }
      if (!text.empty()) {
        std::clog << "Rect: " << bbox.x << ", " << bbox.y << ", " << bbox.width
                  << ", " << bbox.height << "> "
                  << "Recognized text : " << text;
      }
      recognized_texts.push_back(text);
    }

    for (int i = 0; i < detected_boxes.size(); ++i) {
      const auto &bbox = detected_boxes[i];
      cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2);
      cv::putText(image, recognized_texts[i], cv::Point(bbox.x, bbox.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite("vis_ocr_ret.png", image);

  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what();
    return -1;
  }

  return 0;
}
