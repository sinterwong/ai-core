#include "ocr_utils.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include <opencv2/imgcodecs.hpp>

namespace ai_core::example {

OCRUtils *OCRUtils::instance(const std::string &det_config_path,
                             const std::string &rec_config_path,
                             const std::string &dict_path) {

  static OCRUtils instance(det_config_path, rec_config_path, dict_path);
  return &instance;
}

OCRUtils::OCRUtils(const std::string &det_config_path,
                   const std::string &rec_config_path,
                   const std::string &dict_path) {
  try {
    m_ocrDetector = std::make_unique<GenericImageInfer>(det_config_path);
    m_ocrRec = std::make_unique<OCRRec>(rec_config_path, dict_path);
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Failed to initialize OCR: " << e.what();
  }
  LOG_INFO_S << "OCRUtils initialized successfully.";
}

std::vector<std::string> OCRUtils::process(const cv::Mat &frame,
                                           const cv::Rect &roi,
                                           bool need_merge_row,
                                           float expand_ratio_x,
                                           float expand_ratio_y) {
  if (!m_ocrDetector || !m_ocrRec) {
    LOG_ERROR_S << "OCR not initialized.";
    return {};
  }

  std::vector<std::string> recognized_texts;
  std::vector<cv::Rect> detected_boxes = detect(frame, roi);

  if (need_merge_row) {
    detected_boxes = mergeRowBoxes(detected_boxes);
  }

  for (cv::Rect bbox : detected_boxes) {
    if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
      LOG_WARNING_S
          << "Detected bounding box is empty or has zero dimension, skipping.";
      continue;
    }

    bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
      LOG_WARNING_S << "Bounding box clipped to image boundaries is empty or "
                       "has zero dimension, skipping.";
      continue;
    }

    bbox = expandBox(bbox, expand_ratio_x, expand_ratio_y, frame.size());

    if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
      LOG_WARNING_S << "Expanded bounding box is empty or has zero dimension, "
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
        recognized_texts.push_back(text);
      }
    }
  }
  return recognized_texts;
}

std::vector<cv::Rect> OCRUtils::detect(const cv::Mat &frame,
                                       const cv::Rect &roi) {
  ai_core::AlgoOutput algo_output = (*m_ocrDetector)(frame, roi);
  auto ocr_det_ret = algo_output.getParams<ai_core::SegRet>();
  if (ocr_det_ret == nullptr) {
    LOG_ERROR_S << "OCR Detector output is null.";
    return {};
  }

  if (ocr_det_ret->cls_to_contours.size() != 1) {
    LOG_ERROR_S << "OCR Detector output does not contain expected number of "
                   "classes. Expected 1, got "
                << ocr_det_ret->cls_to_contours.size();
    return {};
  }

  const auto &det_contour_rets = ocr_det_ret->cls_to_contours.at(1);
  std::vector<cv::Rect> bboxes;
  for (const auto &contour : det_contour_rets) {
    std::vector<cv::Point> cv_contour;
    cv_contour.reserve(contour.size());
    for (const auto &pt : contour) {
      cv_contour.push_back(ai_core::interop::toCv(pt));
    }
    bboxes.push_back(cv::boundingRect(cv_contour));
  }
  return bboxes;
}

std::vector<std::string> OCRUtils::recognize(const cv::Mat &image_gray) {
  std::vector<cv::Mat> lines = lineSplit(image_gray);
  std::vector<std::string> recognized_texts;
  for (const auto &lineImage : lines) {
    ai_core::OCRRecoRet recoRet = m_ocrRec->process(lineImage);
    recognized_texts.push_back(m_ocrRec->mapToString(recoRet.outputs));
  }
  return recognized_texts;
}

std::vector<cv::Rect> OCRUtils::mergeRowBoxes(std::vector<cv::Rect> boxes) {
  if (boxes.size() <= 1) {
    return boxes;
  }

  std::sort(boxes.begin(), boxes.end(),
            [](const cv::Rect &a, const cv::Rect &b) { return a.y < b.y; });

  std::vector<cv::Rect> merged_boxes;
  cv::Rect current_merged_box = boxes[0];

  for (size_t i = 1; i < boxes.size(); ++i) {
    int current_center_y = current_merged_box.y + current_merged_box.height / 2;
    const cv::Rect &next_box = boxes[i];

    if (next_box.y < current_center_y &&
        current_center_y < (next_box.y + next_box.height)) {
      int new_x = std::min(current_merged_box.x, next_box.x);
      int new_y = std::min(current_merged_box.y, next_box.y);
      int new_width = std::max(current_merged_box.x + current_merged_box.width,
                               next_box.x + next_box.width) -
                      new_x;
      int new_height =
          std::max(current_merged_box.y + current_merged_box.height,
                   next_box.y + next_box.height) -
          new_y;

      current_merged_box = cv::Rect(new_x, new_y, new_width, new_height);
    } else {
      merged_boxes.push_back(current_merged_box);
      current_merged_box = next_box;
    }
  }
  merged_boxes.push_back(current_merged_box);
  return merged_boxes;
}

std::vector<std::pair<cv::Rect, std::string>> OCRUtils::regionsHaveKeywords(
    const cv::Mat &frame, const std::vector<cv::Rect> &rois,
    const std::vector<std::string> &keywords, bool need_merge_row,
    float expand_ratio_x, float expand_ratio_y) {

  std::vector<std::pair<cv::Rect, std::string>> result;
  if (!m_ocrDetector || !m_ocrRec) {
    LOG_ERROR_S << "OCR not initialized.";
    return result;
  }

  for (const auto &roi : rois) {
    std::vector<cv::Rect> detected_boxes = detect(frame, roi);

    if (need_merge_row) {
      detected_boxes = mergeRowBoxes(detected_boxes);
    }

    for (cv::Rect bbox : detected_boxes) {
      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNING_S
            << "Detected bounding box is empty or has zero dimension, "
               "skipping.";
        continue;
      }

      bbox = expandBox(bbox, expand_ratio_x, expand_ratio_y, frame.size());

      if (bbox.empty() || bbox.width == 0 || bbox.height == 0) {
        LOG_WARNING_S
            << "Expanded bounding box is empty or has zero dimension, "
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

bool OCRUtils::hasKeyword(const std::string &ocr_ret,
                          const std::string &keyword) {
  return ocr_ret.find(keyword) != std::string::npos;
}

cv::Rect OCRUtils::expandBox(const cv::Rect &box, float expand_ratio_x,
                             float expand_ratio_y, const cv::Size &frame_size) {
  int expanded_x = static_cast<int>(box.x - box.width * expand_ratio_x / 2);
  int expanded_y = static_cast<int>(box.y - box.height * expand_ratio_y / 2);
  int expanded_width = static_cast<int>(box.width * (1 + expand_ratio_x));
  int expanded_height = static_cast<int>(box.height * (1 + expand_ratio_y));

  cv::Rect expanded_box =
      cv::Rect(expanded_x, expanded_y, expanded_width, expanded_height);
  return expanded_box & cv::Rect(0, 0, frame_size.width, frame_size.height);
}

cv::Mat OCRUtils::convertToBlackWords(const cv::Mat &gray_image) {
  cv::Mat ret_mat = gray_image;
  if (cv::sum(gray_image.col(0))[0] < 145)
    cv::subtract(255, gray_image, ret_mat);
  return ret_mat;
}

/**
 * @brief 将包含多行文本的图像分割成单行文本图像列表。
 * @param grayImage 输入的图像是灰度图
 * @return 包含所有分割出的单行图像的向量。
 */
std::vector<cv::Mat> OCRUtils::lineSplit(const cv::Mat &gray_image) {
  if (gray_image.empty() || gray_image.channels() != 1) {
    LOG_ERROR_S << "Input image is empty or not a single channel image.";
    return {};
  }

  // 转换成黑底白字
  cv::Mat in_img = convertToBlackWords(gray_image);

  constexpr int intensity_threshold = 100;
  // 有效文本行的最小高度
  constexpr int min_line_height = 8;

  std::vector<cv::Mat> result_lines;
  const auto image_rows = in_img.rows;
  const auto image_cols = in_img.cols;

  // 黑字白底 -> 白字黑底
  cv::Mat inverted_img;
  cv::subtract(255, in_img, inverted_img);

  // 沿水平方向进行reduce操作，计算每行的最大像素值，生成垂直投影
  cv::Mat vertical_projection;
  cv::reduce(inverted_img, vertical_projection, 1, cv::REDUCE_MAX);

  // 裁剪逻辑
  auto extract_and_push_line = [&](int start_row, int end_row) {
    if (end_row - start_row + 1 >= min_line_height) {
      int padded_start = std::max(0, start_row - 1);
      int padded_end = std::min(image_rows - 1, end_row + 1);

      cv::Rect roi(0, padded_start, image_cols, padded_end - padded_start + 1);
      result_lines.push_back(in_img(roi).clone());
    }
  };

  int current_line_start = -1;

  // 遍历垂直投影，寻找文本行的起止位置
  for (int i = 0; i < image_rows; ++i) {
    bool is_text_row = vertical_projection.at<uchar>(i) > intensity_threshold;

    if (is_text_row && current_line_start == -1) {
      current_line_start = i;
    } else if (!is_text_row && current_line_start != -1) {
      extract_and_push_line(current_line_start, i - 1);
      current_line_start = -1;
    }
  }

  if (current_line_start != -1) {
    extract_and_push_line(current_line_start, image_rows - 1);
  }

  return result_lines;
}
} // namespace ai_core::example