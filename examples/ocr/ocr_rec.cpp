#include "ocr_rec.hpp"
#include "algo_config_parser.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <logger.hpp>
#include <memory>
#include <opencv2/opencv.hpp>

namespace us_pipe {
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

    std::wifstream dictFile(dictPath);
    dictFile.imbue(std::locale(
        dictFile.getloc(),
        new std::codecvt_utf8_utf16<wchar_t, 0x10ffff, std::consume_header>));

    std::wstring lineTemp;
    while (std::getline(dictFile, lineTemp)) {
      mDict.push_back(lineTemp[0]);
    }
    dictFile.close();
  }

  try {
    mParams = us_pipe::utils::AlgoConfigParser(configPath).parse();
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load OCR config: " +
                             std::string(e.what()));
  }

  const auto &preprocModule = mParams.getParam<std::string>("preprocType");
  const auto &inferModule = mParams.getParam<std::string>("inferType");
  const auto &postprocModule = mParams.getParam<std::string>("postprocType");

  mFramePreproc = std::make_shared<ai_core::dnn::AlgoPreproc>(preprocModule);
  mOcrPostproc = std::make_shared<ai_core::dnn::AlgoPostproc>(postprocModule);

  ai_core::AlgoInferParams inferParams =
      mParams.getParam<ai_core::AlgoInferParams>("inferParams");
  inferParams.decryptkeyStr = SECURITY_KEY;

  mEngine =
      std::make_shared<ai_core::dnn::AlgoInferEngine>(inferModule, inferParams);

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
  ai_core::FramePreprocessArg framePreprocessArg =
      mParams.getParam<ai_core::FramePreprocessArg>("preprocParams");

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

  ai_core::AlgoPostprocParams postprocParams;
  const ai_core::GenericPostParams &genericPostParams =
      mParams.getParam<ai_core::GenericPostParams>("postprocParams");
  postprocParams.setParams(genericPostParams);

  ai_core::AlgoOutput algoOutput;
  if (mOcrPostproc->process(modelOutput, postprocParams, algoOutput,
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

  std::wstring wRet;
  for (int64_t index : recResult) {
    if (index >= 0 && index < mDict.size()) {
      wRet += mDict[index];
    }

    if (index > static_cast<int64_t>(mDict.size())) {
      LOG_ERRORS << "Index out of dictionary bounds: " << index;
      throw std::runtime_error("Index out of dictionary bounds");
    }
  }

  std::string ret;
  try {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    ret = converter.to_bytes(wRet);
  } catch (const std::range_error &e) {
    LOG_ERRORS << "Failed to convert wstring to string: " << e.what();
    return "";
  }
  return ret;
}
} // namespace us_pipe