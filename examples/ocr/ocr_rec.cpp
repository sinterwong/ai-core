#include "ocr_rec.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include "ai_core/preprocess_types.hpp"
#include "algo_config_parser.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

namespace ai_core::example {
namespace fs = std::filesystem;

OCRRec::OCRRec(const std::string &config_path, const std::string &dict_path) {
  if (!fs::exists(config_path)) {
    LOG_ERROR_S << "config file not found: " << config_path;
    throw std::runtime_error("Config file not found: " + config_path);
  }

  if (!dict_path.empty()) {
    LOG_INFO_S << "Dictionary file found: " << dict_path;
    if (!fs::exists(fs::path(dict_path))) {
      LOG_ERROR_S << "Dictionary file not found: " << dict_path;
      throw std::runtime_error("Dictionary file not found: " + dict_path);
    }

    std::ifstream dict_file(dict_path);
    if (!dict_file.is_open()) {
      LOG_ERROR_S << "Failed to open dictionary file: " << dict_path;
      throw std::runtime_error("Failed to open dictionary file: " + dict_path);
    }

    std::string line;
    while (std::getline(dict_file, line)) {
      if (!line.empty()) {
        size_t char_len = 1; // 默认 ASCII
        // 获取第一个 UTF-8 字符
        unsigned char first_byte = static_cast<unsigned char>(line[0]);
        // UTF-8是变长编码，一个字符占1~4 byte。用首字符判断该字符占用多少byte
        if (first_byte >= 0xF0)
          char_len = 4;
        else if (first_byte >= 0xE0)
          char_len = 3;
        else if (first_byte >= 0xC0)
          char_len = 2;

        if (char_len <= line.length()) {
          mDict.push_back(line.substr(0, char_len));
        }
      }
    }
    dict_file.close();
  }

  try {
    mParams = ai_core::example::utils::AlgoConfigParser(config_path).parse();
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load OCR config: " +
                             std::string(e.what()));
  }

  mFramePreproc = std::make_shared<ai_core::dnn::AlgoPreproc>(
      mParams.modelTypes.preproc_module);
  mOcrPostproc = std::make_shared<ai_core::dnn::AlgoPostproc>(
      mParams.modelTypes.postproc_module);

  mEngine = std::make_shared<ai_core::dnn::AlgoInferEngine>(
      mParams.modelTypes.infer_module, mParams.inferParams);

  if (mEngine->initialize() != ai_core::InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "OCRRec engine initialize failed";
    throw std::runtime_error("OCRRec engine initialize failed");
  }

  // The parser hands back all model input names; the frame preprocessor
  // consumes exactly the first one (the second input, input_lengths, is
  // assembled manually in process()).
  ai_core::AlgoPreprocParams bound_preproc_params;
  auto frame_preprocess_arg_ptr =
      mParams.preproc_params.getParams<ai_core::FramePreprocessArg>();
  if (frame_preprocess_arg_ptr == nullptr) {
    LOG_ERROR_S << "FramePreprocessArg is nullptr";
    throw std::runtime_error("FramePreprocessArg is nullptr");
  }
  auto frame_preprocess_arg = *frame_preprocess_arg_ptr;
  frame_preprocess_arg.input_names = {
      frame_preprocess_arg_ptr->input_names.at(0)};
  bound_preproc_params.setParams(frame_preprocess_arg);

  if (mFramePreproc->initialize(bound_preproc_params) !=
      ai_core::InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "OCRRec preprocessor initialize failed";
    throw std::runtime_error("OCRRec preprocessor initialize failed");
  }

  if (mOcrPostproc->initialize(mParams.postproc_params) !=
      ai_core::InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "OCRRec postprocessor initialize failed";
    throw std::runtime_error("OCRRec postprocessor initialize failed");
  }
}

OCRRec::~OCRRec() {}

ai_core::OCRRecoRet OCRRec::process(const cv::Mat &image_gray) {
  ai_core::AlgoInput algo_input;
  ai_core::FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_gray);
  frame_input.roi = ai_core::Rect{0, 0, image_gray.cols, image_gray.rows};
  algo_input.setParams(frame_input);

  auto runtime_context = std::make_shared<ai_core::RuntimeContext>();
  ai_core::TensorData model_input;
  mFramePreproc->process(algo_input, model_input, runtime_context);

  std::vector<int64_t> input_lengths = {1};
  ai_core::TypedBuffer input_lengths_tensor;
  input_lengths_tensor = ai_core::TypedBuffer::createFromCpu(
      ai_core::DataType::INT64,
      std::vector<uint8_t>(
          reinterpret_cast<const uint8_t *>(input_lengths.data()),
          reinterpret_cast<const uint8_t *>(input_lengths.data()) +
              input_lengths.size() * sizeof(int64_t)));
  const auto &input_names =
      mParams.preproc_params.getParams<ai_core::FramePreprocessArg>()
          ->input_names;
  model_input.set(input_names.at(1), input_lengths_tensor, {1});

  ai_core::TensorData model_output;
  if (mEngine->infer(model_input, model_output) !=
      ai_core::InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "OCRRec engine infer failed";
    return {};
  }

  ai_core::AlgoOutput algo_output;
  if (mOcrPostproc->process(model_output, algo_output, runtime_context) !=
      ai_core::InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "OCRRec postprocess failed";
    return {};
  }

  auto ocr_ret = algo_output.getParams<ai_core::OCRRecoRet>();
  if (ocr_ret == nullptr) {
    LOG_ERROR_S << "OCRRecoRet is nullptr";
    return {};
  }
  return *ocr_ret;
}

std::string OCRRec::mapToString(const std::vector<int64_t> &rec_result) {
  if (mDict.empty()) {
    LOG_WARNING_S
        << "Dictionary is empty, cannot map recognition results to string.";
    return "";
  }

  std::string ret;
  for (int64_t index : rec_result) {
    if (index >= 0 && index < static_cast<int64_t>(mDict.size())) {
      ret += mDict[index];
    } else if (index > static_cast<int64_t>(mDict.size())) {
      LOG_ERROR_S << "Index out of dictionary bounds: " << index;
      throw std::runtime_error("Index out of dictionary bounds");
    }
  }
  return ret;
}
} // namespace ai_core::example