/**
 * @file ocr_main.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-09-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/logger.hpp"
#include "ocr_utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <config_path> <image_path> <dict_path>" << std::endl;
    return -1;
  }

  std::string det_config_path = argv[1];
  std::string rec_config_path = argv[2];
  std::string image_path = argv[3];
  std::string dict_path = argv[4];

  ai_core::logging::Logger::instance().setLevel(
      ai_core::logging::LogLevel::Trace);
  ai_core::logging::Logger::instance().enableConsole(true);
  ai_core::logging::Logger::instance().enableFile(false);
  ai_core::logging::Logger::instance().enableColor(true);

  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    LOG_ERROR_S << "Failed to read image: " << image_path;
    return -1;
  }

  try {
    ai_core::example::OCRUtils *ocr = ai_core::example::OCRUtils::instance(
        det_config_path, rec_config_path, dict_path);

    auto detected_boxes = ocr->detect(image);
    LOG_INFO_S << "Detected BBoxes: " << detected_boxes.size();

    std::vector<std::string> recognized_texts;
    for (auto &bbox : detected_boxes) {
      bbox = ocr->expandBox(bbox, 0.0f, 0.5f, image.size());
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNING_S
            << "Expanded bounding box is empty or has zero dimension, "
               "skipping.";
        continue;
      }
      // make sure the bbox is inside the image
      bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNING_S << "Expanded bounding box clipped to image boundaries is "
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
        LOG_INFO_S << "Rect: " << bbox.x << ", " << bbox.y << ", " << bbox.width
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
    LOG_ERROR_S << "An error occurred: " << e.what();
    return -1;
  }

  return 0;
}
