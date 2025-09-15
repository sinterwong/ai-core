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
#include "ocr_rec.hpp"
#include <iostream>
#include <logger.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <config_path> <image_path> <dict_path>" << std::endl;
    return -1;
  }

  std::string configPath = argv[1];
  std::string imagePath = argv[2];
  std::string dictPath = argv[3];

  Logger::LogConfig logConfig;
  logConfig.appName = "OCR-Main";
  logConfig.logPath = "./logs";
  logConfig.logLevel = LogLevel::INFO;
  logConfig.enableConsole = true;
  logConfig.enableColor = true;
  Logger::instance()->initialize(logConfig);

  cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    LOG_ERRORS << "Failed to read image: " << imagePath;
    return -1;
  }

  try {
    us_pipe::OCRRec ocr(configPath, dictPath);
    ai_core::OCRRecoRet result = ocr.process(image);

    if (result.outputs.empty()) {
      LOG_INFOS << "No text recognized.";
    } else {
      std::string recognizedText = ocr.mapToString(result.outputs);
      LOG_INFOS << "Recognized Text: " << recognizedText;
    }

  } catch (const std::exception &e) {
    LOG_ERRORS << "OCR process failed: " << e.what();
    return -1;
  }
  return 0;
}
