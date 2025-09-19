/**
 * @file yoloDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "yolo_det.hpp"
#include "logger.hpp"
#include "vision_util.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool Yolov11Det::process(const TensorData &modelOutput,
                         const FrameTransformContext &prepArgs,
                         const AnchorDetParams &postArgs,
                         AlgoOutput &algoOutput) const {
  const auto &outputShapes = modelOutput.shapes;
  const auto &inputShape = prepArgs.modelInputShape;
  const auto &outputs = modelOutput.datas;

  // just one output
  if (outputs.size() != 1) {
    LOG_ERRORS << "AnchorDetParams(Yolov11Det) unexpected size of outputs "
               << outputs.size();
    throw std::runtime_error(
        "AnchorDetParams(Yolov11Det)  unexpected size of outputs");
  }
  auto output = outputs.at(postArgs.outputNames.at(0));

  std::vector<int> outputShape = outputShapes.at(postArgs.outputNames.at(0));
  int signalResultNum = outputShape.at(outputShape.size() - 2);
  int strideNum = outputShape.at(outputShape.size() - 1);

  cv::Mat rawData = cv::Mat(strideNum, signalResultNum, CV_32F);
  if (output.dataType() == DataType::FLOAT32) {
    cv::transpose(cv::Mat(signalResultNum, strideNum, CV_32F,
                          const_cast<void *>(output.getRawHostPtr())),
                  rawData);
  } else if (output.dataType() == DataType::FLOAT16) {
    const uint16_t *fp16Data = output.getHostPtr<uint16_t>();
    cv::Mat halfMat(1, output.getElementCount(), CV_16F, (void *)fp16Data);
    cv::Mat floatMat(1, output.getElementCount(), CV_32F);
    halfMat.convertTo(floatMat, CV_32F);
    cv::transpose(cv::Mat(signalResultNum, strideNum, CV_32F, floatMat.data),
                  rawData);
  }

  std::vector<BBox> results = processRawOutput(rawData, inputShape, prepArgs,
                                               postArgs, signalResultNum - 4);

  DetRet detRet;
  detRet.bboxes = utils::NMS(results, postArgs.nmsThre, postArgs.condThre);
  algoOutput.setParams(detRet);
  return true;
}

bool Yolov11Det::batchProcess(
    const TensorData &modelOutput,
    const std::vector<FrameTransformContext> &prepArgs,
    const AnchorDetParams &postArgs,
    std::vector<AlgoOutput> &algoOutput) const {
  const auto &outputs = modelOutput.datas;
  if (outputs.size() != 1) {
    LOG_ERRORS << "Yolov11Det::batchProcess unexpected size of outputs: "
               << outputs.size();
    throw std::runtime_error(
        "Yolov11Det::batchProcess expects only 1 output tensor.");
  }
  const auto &outputTensor = outputs.at(postArgs.outputNames.at(0));
  const auto &outputShape = modelOutput.shapes.at(postArgs.outputNames.at(0));

  if (outputShape.size() != 3) {
    LOG_ERRORS << "Yolov11Det::batchProcess unexpected output dimensions: "
               << outputShape.size();
    throw std::runtime_error("Yolov11Det::batchProcess expects a 3D output "
                             "tensor [batch, stride, num_results].");
  }
  const int batchSize = outputShape.at(0);
  const int strideNum = outputShape.at(1);
  const int signalResultNum = outputShape.at(2);
  const int numClasses = strideNum - 4;

  if (batchSize != prepArgs.size()) {
    LOG_ERRORS
        << "Yolov11Det::batchProcess mismatch between model output batch size ("
        << batchSize << ") and prepArgs size (" << prepArgs.size() << ").";
    throw std::runtime_error(
        "Batch size mismatch in Yolov11Det::batchProcess.");
  }

  algoOutput.resize(batchSize);
  const size_t elementsPerSample =
      static_cast<size_t>(strideNum) * signalResultNum;

  const float *batchedFloatData = nullptr;
  cv::Mat fullFloatMat;
  if (outputTensor.dataType() == DataType::FLOAT32) {
    batchedFloatData = outputTensor.getHostPtr<float>();
  } else if (outputTensor.dataType() == DataType::FLOAT16) {
    const uint16_t *fp16Data = outputTensor.getHostPtr<uint16_t>();
    cv::Mat halfMat(1, outputTensor.getElementCount(), CV_16F,
                    const_cast<uint16_t *>(fp16Data));
    halfMat.convertTo(fullFloatMat, CV_32F);
    batchedFloatData = fullFloatMat.ptr<float>();
  } else {
    throw std::runtime_error(
        "Unsupported data type in Yolov11Det::batchProcess.");
  }

  for (int i = 0; i < batchSize; ++i) {
    const float *currentSampleData = batchedFloatData + i * elementsPerSample;

    cv::Mat singleOutputMat(strideNum, signalResultNum, CV_32F,
                            const_cast<float *>(currentSampleData));
    cv::Mat transposedOutput;
    cv::transpose(singleOutputMat, transposedOutput);

    const auto &currentPrepArgs = prepArgs[i];
    const auto &inputShape = currentPrepArgs.modelInputShape;

    std::vector<BBox> results = processRawOutput(
        transposedOutput, inputShape, currentPrepArgs, postArgs, numClasses);

    DetRet detRet;
    detRet.bboxes = utils::NMS(results, postArgs.nmsThre, postArgs.condThre);
    algoOutput[i].setParams(detRet);
  }
  return true;
}

std::vector<BBox> Yolov11Det::processRawOutput(
    const cv::Mat &transposedOutput, const Shape &inputShape,
    const FrameTransformContext &prepArgs, const AnchorDetParams &postArgs,
    int numClasses) const {
  std::vector<BBox> results;
  Shape originShape;
  const auto &inputRoi = *prepArgs.roi;
  if (inputRoi.area() > 0) {
    originShape.w = inputRoi.width;
    originShape.h = inputRoi.height;
  } else {
    originShape = prepArgs.originShape;
  }
  auto [scaleX, scaleY] =
      utils::scaleRatio(originShape, inputShape, prepArgs.isEqualScale);

  for (int i = 0; i < transposedOutput.rows; ++i) {
    const float *data = transposedOutput.ptr<float>(i);

    cv::Mat scores(1, numClasses, CV_32F, (void *)(data + 4));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

    if (score > postArgs.condThre) {
      BBox result;
      result.score = score;
      result.label = classIdPoint.x;

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      x = x - 0.5 * w;
      y = y - 0.5 * h;

      if (prepArgs.isEqualScale) {
        x = (x - prepArgs.leftPad) / scaleX;
        y = (y - prepArgs.topPad) / scaleY;
      } else {
        x = x / scaleX;
        y = y / scaleY;
      }
      w = w / scaleX;
      h = h / scaleY;
      x += inputRoi.x;
      y += inputRoi.y;
      result.rect =
          std::make_shared<cv::Rect>(static_cast<int>(x), static_cast<int>(y),
                                     static_cast<int>(w), static_cast<int>(h));

      results.push_back(result);
    }
  }

  return results;
}
} // namespace ai_core::dnn
