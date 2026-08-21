#ifndef AI_CORE_CPU_GENERIC_PREPROCESS_HPP
#define AI_CORE_CPU_GENERIC_PREPROCESS_HPP

#include "cpu_generic_preprocessor.hpp"
#include "preproc/generic_frame_preproc_base.hpp"

namespace ai_core::dnn {

class CpuGenericPreprocess final : public GenericFramePreprocBase {
protected:
  const IFramePreprocessor &kernel() const override { return m_kernel; }

private:
  cpu::CpuGenericCvPreprocessor m_kernel;
};

} // namespace ai_core::dnn

#endif
