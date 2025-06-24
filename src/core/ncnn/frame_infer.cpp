/**
 * @file dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "frame_infer.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core/mat.hpp>

namespace infer::dnn {

std::vector<std::pair<std::string, ncnn::Mat>>
FrameInference::preprocess(AlgoInput &input) const {
  // Get input parameters
  auto *frameInput = input.getParams<FrameInput>();
  if (!frameInput) {
    LOG_ERRORS << "Invalid input parameters";
    throw std::runtime_error("Invalid input parameters");
  }

  int inputWidth = params->inputShape.w;
  int inputHeight = params->inputShape.h;
  auto &args = frameInput->args;
  const cv::Mat &image = frameInput->image;

  ncnn::Mat::PixelType pixelType =
      image.channels() == 1 ? ncnn::Mat::PIXEL_GRAY : ncnn::Mat::PIXEL_RGB;

  // crop roi
  cv::Mat croppedImage;
  if (args.roi.area() > 0)
    croppedImage = image(args.roi).clone();
  else
    croppedImage = image;

  // resize
  ncnn::Mat in;
  if (args.needResize) {
    if (args.isEqualScale) {
      cv::Mat resizedImage;
      auto padRet = utils::escaleResizeWithPad(
          croppedImage, resizedImage, inputHeight, inputHeight, args.pad);
      args.topPad = padRet.h;
      args.leftPad = padRet.w;
      in = ncnn::Mat::from_pixels(resizedImage.data, pixelType,
                                  resizedImage.cols, resizedImage.rows);
    } else {
      in = ncnn::Mat::from_pixels_resize(croppedImage.data, pixelType,
                                         croppedImage.cols, croppedImage.rows,
                                         inputWidth, inputHeight);
    }
  } else {
    int depth = croppedImage.depth();
    if (depth == CV_8U) {
      in = ncnn::Mat::from_pixels(croppedImage.data, pixelType,
                                  croppedImage.cols, croppedImage.rows);
    } else if (depth == CV_32F) {
      in = ncnn::Mat(croppedImage.cols, croppedImage.rows, 1, sizeof(float));
      memcpy(in.data, croppedImage.data, croppedImage.total() * sizeof(float));
    } else {
      throw std::runtime_error("Unsupported image depth");
    }
  }

  // normalize
  if (!args.normVals.empty() || !args.meanVals.empty()) {
    std::vector<float> normVals;
    std::transform(args.normVals.begin(), args.normVals.end(),
                   std::back_inserter(normVals),
                   [](float val) { return 1.0f / val; });
    in.substract_mean_normalize(args.meanVals.data(), normVals.data());
  }
  return {{inputNames[0], in}};
}
}; // namespace infer::dnn
