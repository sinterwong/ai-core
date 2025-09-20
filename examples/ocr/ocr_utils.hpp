#pragma once
#include "generic_image_infer.hpp"
#include "ocr_rec.hpp"
#include <string>

namespace ai_core::example {

class OCRUtils {
public:
  static OCRUtils *instance(const std::string &detConfigPath,
                            const std::string &recConfigPath,
                            const std::string &dictPath = "");

  ~OCRUtils() {}

  std::vector<std::string> process(const cv::Mat &frame,
                                   const cv::Rect &roi = cv::Rect{},
                                   bool needMergeRow = true,
                                   float expandRatioX = 0.f,
                                   float expandRatioY = 0.2f);

  std::vector<std::pair<cv::Rect, std::string>>
  regionsHaveKeywords(const cv::Mat &frame, const std::vector<cv::Rect> &rois,
                      const std::vector<std::string> &keywords,
                      bool needMergeRow = true, float expandRatioX = 0.f,
                      float expandRatioY = 0.1f);

  std::vector<cv::Rect> detect(const cv::Mat &frame,
                               const cv::Rect &roi = cv::Rect{});

  std::string recognize(const cv::Mat &imageGray);

  cv::Rect expandBox(const cv::Rect &box, float expandRatioX,
                     float expandRatioY, const cv::Size &frameSize);

private:
  std::vector<cv::Rect> mergeRowBoxes(std::vector<cv::Rect> boxes);

  bool hasKeyword(const std::string &ocrRet, const std::string &keyword);

private:
  std::unique_ptr<GenericImageInfer> m_ocrDetector;
  std::unique_ptr<OCRRec> m_ocrRec;

private:
  OCRUtils(const OCRUtils &) = delete;
  OCRUtils &operator=(const OCRUtils &) = delete;
  OCRUtils(OCRUtils &&) = delete;
  OCRUtils &operator=(OCRUtils &&) = delete;

  OCRUtils(const std::string &detConfigPath, const std::string &recConfigPath,
           const std::string &dictPath = "");
};

} // namespace ai_core::example