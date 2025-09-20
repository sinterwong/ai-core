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
#include "ai_core/algo_output_types.hpp"
#include "ocr_utils.hpp"
#include <iostream>
#include <logger.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <config_path> <image_path> <dict_path>" << std::endl;
    return -1;
  }

  std::string detConfigPath = argv[1];
  std::string recConfigPath = argv[2];
  std::string imagePath = argv[3];
  std::string dictPath = argv[4];

  Logger::LogConfig logConfig;
  logConfig.appName = "OCR-Main";
  logConfig.logPath = "./logs";
  logConfig.logLevel = LogLevel::INFO;
  logConfig.enableConsole = true;
  logConfig.enableColor = true;
  Logger::instance()->initialize(logConfig);

  cv::Mat image = cv::imread(imagePath);

  if (image.empty()) {
    LOG_ERRORS << "Failed to read image: " << imagePath;
    return -1;
  }

  try {
    ai_core::example::OCRUtils *ocr = ai_core::example::OCRUtils::instance(
        detConfigPath, recConfigPath, dictPath);

    auto detectedBBoxes = ocr->detect(image);
    LOG_INFOS << "Detected BBoxes: " << detectedBBoxes.size();

    std::vector<std::string> recognizedTexts;
    for (auto &bbox : detectedBBoxes) {
      bbox = ocr->expandBox(bbox, 0.0f, 0.5f, image.size());
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNINGS << "Expanded bounding box is empty or has zero dimension, "
                        "skipping.";
        continue;
      }
      // make sure the bbox is inside the image
      bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNINGS << "Expanded bounding box clipped to image boundaries is "
                        "empty or has zero dimension, skipping.";
        continue;
      }

      cv::Mat textImage = image(bbox);
      cv::Mat grayImage;
      if (textImage.channels() == 3) {
        cv::cvtColor(textImage, grayImage, cv::COLOR_RGB2GRAY);
      } else {
        grayImage = textImage;
      }
      std::string text = ocr->recognize(grayImage);
      if (!text.empty()) {
        LOG_INFOS << "Rect: " << bbox.x << ", " << bbox.y << ", " << bbox.width
                  << ", " << bbox.height << "> "
                  << "Recognized text : " << text;
      }
      recognizedTexts.push_back(text);
    }

    for (int i = 0; i < detectedBBoxes.size(); ++i) {
      const auto &bbox = detectedBBoxes[i];
      cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2);
      cv::putText(image, recognizedTexts[i], cv::Point(bbox.x, bbox.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite("vis_ocr_ret.png", image);

  } catch (const std::exception &e) {
    LOG_ERRORS << "An error occurred: " << e.what();
    return -1;
  }

  return 0;
}
