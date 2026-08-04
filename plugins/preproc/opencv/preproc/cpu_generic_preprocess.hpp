/**
 * @file cpu_generic_preprocess.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Generic frame preprocessing plugin backed by the OpenCV CPU kernel
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_CPU_GENERIC_PREPROCESS_HPP
#define AI_CORE_CPU_GENERIC_PREPROCESS_HPP

#include "cpu_generic_preprocessor.hpp"
#include "generic_frame_preproc_base.hpp"

namespace ai_core::dnn {

class CpuGenericPreprocess final : public GenericFramePreprocBase {
protected:
  const IFramePreprocessor &kernel() const override { return m_kernel; }

private:
  cpu::CpuGenericCvPreprocessor m_kernel;
};

} // namespace ai_core::dnn

#endif
