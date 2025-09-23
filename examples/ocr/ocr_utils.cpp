#include "ocr_utils.hpp"
#include <logger.hpp>
#include <opencv2/imgcodecs.hpp>

namespace ai_core::example {

OCRUtils *OCRUtils::instance(const std::string &detConfigPath,
                             const std::string &recConfigPath,
                             const std::string &dictPath) {

  static OCRUtils instance(detConfigPath, recConfigPath, dictPath);
  return &instance;
}

OCRUtils::OCRUtils(const std::string &detConfigPath,
                   const std::string &recConfigPath,
                   const std::string &dictPath) {
  try {
    m_ocrDetector = std::make_unique<GenericImageInfer>(detConfigPath);
    m_ocrRec = std::make_unique<OCRRec>(recConfigPath, dictPath);
  } catch (const std::exception &e) {
    LOG_ERRORS << "Failed to initialize OCR: " << e.what();
  }
  LOG_INFOS << "OCRUtils initialized successfully.";
}

std::vector<std::string>
OCRUtils::process(const cv::Mat &frame, const cv::Rect &roi, bool needMergeRow,
                  float expandRatioX, float expandRatioY) {
  if (!m_ocrDetector || !m_ocrRec) {
    LOG_ERRORS << "OCR not initialized.";
    return {};
  }

  std::vector<std::string> recognizedTexts;
  std::vector<cv::Rect> detectedBBoxes = detect(frame, roi);

  if (needMergeRow) {
    detectedBBoxes = mergeRowBoxes(detectedBBoxes);
  }

  for (cv::Rect bbox : detectedBBoxes) {
    if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
      LOG_WARNINGS
          << "Detected bounding box is empty or has zero dimension, skipping.";
      continue;
    }

    bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
      LOG_WARNINGS << "Bounding box clipped to image boundaries is empty or "
                      "has zero dimension, skipping.";
      continue;
    }

    bbox = expandBox(bbox, expandRatioX, expandRatioY, frame.size());

    if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
      LOG_WARNINGS << "Expanded bounding box is empty or has zero dimension, "
                      "skipping.";
      continue;
    }

    cv::Mat textImage = frame(bbox);
    // cv::imwrite("text_image.png", textImage);
    cv::Mat grayImage;
    if (textImage.channels() == 3) {
      cv::cvtColor(textImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
      grayImage = textImage;
    }

    std::vector<std::string> texts = recognize(grayImage);
    if (!texts.empty()) {
      for (const auto &text : texts) {
        recognizedTexts.push_back(text);
      }
    }
  }
  return recognizedTexts;
}

std::vector<cv::Rect> OCRUtils::detect(const cv::Mat &frame,
                                       const cv::Rect &roi) {
  ai_core::AlgoOutput algo_output = (*m_ocrDetector)(frame, roi);
  auto ocrDetRet = algo_output.getParams<ai_core::SegRet>();
  if (ocrDetRet == nullptr) {
    LOG_ERRORS << "OCR Detector output is null.";
    return {};
  }

  if (ocrDetRet->clsToContours.size() != 1) {
    LOG_ERRORS << "OCR Detector output does not contain expected number of "
                  "classes. Expected 1, got "
               << ocrDetRet->clsToContours.size();
    return {};
  }

  const auto &detContourRets = ocrDetRet->clsToContours.at(1);
  std::vector<cv::Rect> bboxes;
  for (const auto &contour : detContourRets) {
    bboxes.push_back(cv::boundingRect(contour));
  }
  return bboxes;
}

std::vector<std::string> OCRUtils::recognize(const cv::Mat &imageGray) {
  std::vector<cv::Mat> lines = lineSplit(imageGray);
  std::vector<std::string> recognizedTexts;
  for (const auto &lineImage : lines) {
    ai_core::OCRRecoRet recoRet = m_ocrRec->process(lineImage);
    recognizedTexts.push_back(m_ocrRec->mapToString(recoRet.outputs));
  }
  return recognizedTexts;
}

std::vector<cv::Rect> OCRUtils::mergeRowBoxes(std::vector<cv::Rect> boxes) {
  if (boxes.size() <= 1) {
    return boxes;
  }

  std::sort(boxes.begin(), boxes.end(),
            [](const cv::Rect &a, const cv::Rect &b) { return a.y < b.y; });

  std::vector<cv::Rect> mergedBoxes;
  cv::Rect currentMergedBox = boxes[0];

  for (size_t i = 1; i < boxes.size(); ++i) {
    int currentCenterY = currentMergedBox.y + currentMergedBox.height / 2;
    const cv::Rect &nextBox = boxes[i];

    if (nextBox.y < currentCenterY &&
        currentCenterY < (nextBox.y + nextBox.height)) {
      int newX = std::min(currentMergedBox.x, nextBox.x);
      int newY = std::min(currentMergedBox.y, nextBox.y);
      int newWidth = std::max(currentMergedBox.x + currentMergedBox.width,
                              nextBox.x + nextBox.width) -
                     newX;
      int newHeight = std::max(currentMergedBox.y + currentMergedBox.height,
                               nextBox.y + nextBox.height) -
                      newY;

      currentMergedBox = cv::Rect(newX, newY, newWidth, newHeight);
    } else {
      mergedBoxes.push_back(currentMergedBox);
      currentMergedBox = nextBox;
    }
  }
  mergedBoxes.push_back(currentMergedBox);
  return mergedBoxes;
}

std::vector<std::pair<cv::Rect, std::string>> OCRUtils::regionsHaveKeywords(
    const cv::Mat &frame, const std::vector<cv::Rect> &rois,
    const std::vector<std::string> &keywords, bool needMergeRow,
    float expandRatioX, float expandRatioY) {

  std::vector<std::pair<cv::Rect, std::string>> result;
  if (!m_ocrDetector || !m_ocrRec) {
    LOG_ERRORS << "OCR not initialized.";
    return result;
  }

  for (const auto &roi : rois) {
    std::vector<cv::Rect> detectedBBoxes = detect(frame, roi);

    if (needMergeRow) {
      detectedBBoxes = mergeRowBoxes(detectedBBoxes);
    }

    for (cv::Rect bbox : detectedBBoxes) {
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNINGS << "Detected bounding box is empty or has zero dimension, "
                        "skipping.";
        continue;
      }

      bbox = expandBox(bbox, expandRatioX, expandRatioY, frame.size());

      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNINGS << "Expanded bounding box is empty or has zero dimension, "
                        "skipping.";
        continue;
      }

      cv::Mat textImage = frame(bbox);
      cv::imwrite("text_image.png", textImage);
      cv::Mat grayImage;
      if (textImage.channels() == 3) {
        cv::cvtColor(textImage, grayImage, cv::COLOR_BGR2GRAY);
      } else {
        grayImage = textImage;
      }

      std::vector<std::string> texts = recognize(grayImage);
      if (!texts.empty()) {
        for (const auto &text : texts) {
          for (const std::string &keyword : keywords) {
            if (hasKeyword(text, keyword)) {
              result.push_back(std::make_pair(bbox, text));
              break;
            }
          }
        }
      }
    }
  }
  return result;
}

bool OCRUtils::hasKeyword(const std::string &ocrRet,
                          const std::string &keyword) {
  return ocrRet.find(keyword) != std::string::npos;
}

cv::Rect OCRUtils::expandBox(const cv::Rect &box, float expandRatioX,
                             float expandRatioY, const cv::Size &frameSize) {
  int expandedX = static_cast<int>(box.x - box.width * expandRatioX / 2);
  int expandedY = static_cast<int>(box.y - box.height * expandRatioY / 2);
  int expandedWidth = static_cast<int>(box.width * (1 + expandRatioX));
  int expandedHeight = static_cast<int>(box.height * (1 + expandRatioY));

  cv::Rect expandedBox =
      cv::Rect(expandedX, expandedY, expandedWidth, expandedHeight);
  return expandedBox & cv::Rect(0, 0, frameSize.width, frameSize.height);
}

cv::Mat OCRUtils::convertToBlackWords(const cv::Mat &grayImage) {
  cv::Mat retMat = grayImage;
  if (cv::sum(grayImage.col(0))[0] < 145)
    cv::subtract(255, grayImage, retMat);
  return retMat;
}

/**
 * @brief 将包含多行文本的图像分割成单行文本图像列表。
 * @param grayImage 输入的图像是灰度图
 * @return 包含所有分割出的单行图像的向量。
 */
std::vector<cv::Mat> OCRUtils::lineSplit(const cv::Mat &grayImage) {
  if (grayImage.empty() || grayImage.channels() != 1) {
    LOG_ERRORS << "Input image is empty or not a single channel image.";
    return {};
  }

  // 转换成黑底白字
  cv::Mat inImg = convertToBlackWords(grayImage);

  constexpr int intensityThreshold = 100;
  // 有效文本行的最小高度
  constexpr int minLineHeight = 8;

  std::vector<cv::Mat> resultLines;
  const auto imageRows = inImg.rows;
  const auto imageCols = inImg.cols;

  // 黑字白底 -> 白字黑底
  cv::Mat invertedImg;
  cv::subtract(255, inImg, invertedImg);

  // 沿水平方向进行reduce操作，计算每行的最大像素值，生成垂直投影
  cv::Mat verticalProjection;
  cv::reduce(invertedImg, verticalProjection, 1, cv::REDUCE_MAX);

  // 裁剪逻辑
  auto extractAndPushLine = [&](int startRow, int endRow) {
    if (endRow - startRow + 1 >= minLineHeight) {
      int paddedStart = std::max(0, startRow - 1);
      int paddedEnd = std::min(imageRows - 1, endRow + 1);

      cv::Rect roi(0, paddedStart, imageCols, paddedEnd - paddedStart + 1);
      resultLines.push_back(inImg(roi).clone());
    }
  };

  int currentLineStart = -1;

  // 遍历垂直投影，寻找文本行的起止位置
  for (int i = 0; i < imageRows; ++i) {
    bool isTextRow = verticalProjection.at<uchar>(i) > intensityThreshold;

    if (isTextRow && currentLineStart == -1) {
      currentLineStart = i;
    } else if (!isTextRow && currentLineStart != -1) {
      extractAndPushLine(currentLineStart, i - 1);
      currentLineStart = -1;
    }
  }

  if (currentLineStart != -1) {
    extractAndPushLine(currentLineStart, imageRows - 1);
  }

  return resultLines;
}
} // namespace ai_core::example