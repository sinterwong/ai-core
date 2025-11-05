#include "ocr_rec.hpp"
#include "ai_core/preproc_types.hpp"
#include "algo_config_parser.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <logger.hpp>
#include <memory>
#include <opencv2/opencv.hpp>

namespace ai_core::example {
namespace fs = std::filesystem;

OCRRec::OCRRec(const std::string &configPath, const std::string &dictPath) {
  if (!fs::exists(configPath)) {
    LOG_ERRORS << "config file not found: " << configPath;
    throw std::runtime_error("Config file not found: " + configPath);
  }

  if (!dictPath.empty()) {
    LOG_INFOS << "Dictionary file found: " << dictPath;
    if (!fs::exists(fs::path(dictPath))) {
      LOG_ERRORS << "Dictionary file not found: " << dictPath;
      throw std::runtime_error("Dictionary file not found: " + dictPath);
    }

    std::ifstream dictFile(dictPath);
    if (!dictFile.is_open()) {
      LOG_ERRORS << "Failed to open dictionary file: " << dictPath;
      throw std::runtime_error("Failed to open dictionary file: " + dictPath);
    }

    std::string line;
    while (std::getline(dictFile, line)) {
      if (!line.empty()) {
        size_t charLen = 1; // 默认 ASCII
        // 获取第一个 UTF-8 字符
        unsigned char firstByte = static_cast<unsigned char>(line[0]);
        // UTF-8是变长编码，一个字符占1~4 byte。用首字符判断该字符占用多少byte
        if (firstByte >= 0xF0)
          charLen = 4;
        else if (firstByte >= 0xE0)
          charLen = 3;
        else if (firstByte >= 0xC0)
          charLen = 2;

        if (charLen <= line.length()) {
          mDict.push_back(line.substr(0, charLen));
        }
      }
    }
    dictFile.close();
  }

  try {
    mParams = ai_core::example::utils::AlgoConfigParser(configPath).parse();
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load OCR config: " +
                             std::string(e.what()));
  }

  mFramePreproc = std::make_shared<ai_core::dnn::AlgoPreproc>(
      mParams.modelTypes.preprocModule);
  mOcrPostproc = std::make_shared<ai_core::dnn::AlgoPostproc>(
      mParams.modelTypes.postprocModule);

  mEngine = std::make_shared<ai_core::dnn::AlgoInferEngine>(
      mParams.modelTypes.inferModule, mParams.inferParams);

  if (mEngine->initialize() != ai_core::InferErrorCode::SUCCESS) {
    LOG_ERRORS << "OCRRec engine initialize failed";
    throw std::runtime_error("OCRRec engine initialize failed");
  }

  if (mFramePreproc->initialize() != ai_core::InferErrorCode::SUCCESS) {
    LOG_ERRORS << "OCRRec preprocessor initialize failed";
    throw std::runtime_error("OCRRec preprocessor initialize failed");
  }

  if (mOcrPostproc->initialize() != ai_core::InferErrorCode::SUCCESS) {
    LOG_ERRORS << "OCRRec postprocessor initialize failed";
    throw std::runtime_error("OCRRec postprocessor initialize failed");
  }
}

OCRRec::~OCRRec() {}

ai_core::OCRRecoRet OCRRec::process(const cv::Mat &imageGray) {
  ai_core::AlgoPreprocParams preprocParams;

  auto framePreprocessArgPtr =
      mParams.preprocParams.getParams<ai_core::FramePreprocessArg>();

  if (framePreprocessArgPtr == nullptr) {
    LOG_ERRORS << "FramePreprocessArg is nullptr";
    throw std::runtime_error("FramePreprocessArg is nullptr");
  }

  auto framePreprocessArg = *framePreprocessArgPtr;

  auto inputNames = framePreprocessArg.inputNames;
  framePreprocessArg.inputNames = {inputNames.at(0)};
  preprocParams.setParams(framePreprocessArg);

  ai_core::AlgoInput algoInput;
  ai_core::FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageGray);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, imageGray.cols, imageGray.rows);
  algoInput.setParams(frameInput);

  auto runtimeContext = std::make_shared<ai_core::RuntimeContext>();
  ai_core::TensorData modelInput;
  mFramePreproc->process(algoInput, preprocParams, modelInput, runtimeContext);

  std::vector<int64_t> inputLengths = {1};
  ai_core::TypedBuffer inputLengthsTensor;
  inputLengthsTensor.setCpuData(
      ai_core::DataType::INT64,
      std::vector<uint8_t>(
          reinterpret_cast<const uint8_t *>(inputLengths.data()),
          reinterpret_cast<const uint8_t *>(inputLengths.data()) +
              inputLengths.size() * sizeof(int64_t)));
  modelInput.datas.insert({inputNames.at(1), inputLengthsTensor});
  modelInput.shapes.insert({inputNames.at(1), {1}});

  ai_core::TensorData modelOutput;
  if (mEngine->infer(modelInput, modelOutput) !=
      ai_core::InferErrorCode::SUCCESS) {
    LOG_ERRORS << "OCRRec engine infer failed";
    return {};
  }

  ai_core::AlgoOutput algoOutput;
  if (mOcrPostproc->process(modelOutput, mParams.postprocParams, algoOutput,
                            runtimeContext) !=
      ai_core::InferErrorCode::SUCCESS) {
    LOG_ERRORS << "OCRRec postprocess failed";
    return {};
  }

  auto ocrRet = algoOutput.getParams<ai_core::OCRRecoRet>();
  if (ocrRet == nullptr) {
    LOG_ERRORS << "OCRRecoRet is nullptr";
    return {};
  }
  return *ocrRet;
}

std::string OCRRec::mapToString(const std::vector<int64_t> &recResult) {
  if (mDict.empty()) {
    LOG_WARNINGS
        << "Dictionary is empty, cannot map recognition results to string.";
    return "";
  }

  std::string ret;
  for (int64_t index : recResult) {
    if (index >= 0 && index < static_cast<int64_t>(mDict.size())) {
      ret += mDict[index];
    } else if (index > static_cast<int64_t>(mDict.size())) {
      LOG_ERRORS << "Index out of dictionary bounds: " << index;
      throw std::runtime_error("Index out of dictionary bounds");
    }
  }
  return ret;
}
} // namespace ai_core::example