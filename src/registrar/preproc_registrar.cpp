/**
 * @file preproc_registrar.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "preproc_registrar.hpp"
#include "logger.hpp"
#include "preproc/frame_prep.hpp"

namespace ai_core::dnn {
#define REGISTER_PREPROCESS_ALGO(AlgoName)                                     \
  PreprocFactory::instance().registerCreator(                                  \
      #AlgoName,                                                               \
      [](const AlgoConstructParams &cparams)                                   \
          -> std::shared_ptr<PreprocssBase> {                                  \
        return std::make_shared<AlgoName>();                                   \
      });                                                                      \
  LOG_INFOS << "Registered " #AlgoName " creator."

PreprocessRegistrar::PreprocessRegistrar() {
  REGISTER_PREPROCESS_ALGO(FramePreprocess);
}
} // namespace ai_core::dnn
