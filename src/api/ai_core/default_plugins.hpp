/**
 * @file default_plugins.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Explicit registration entry for the built-in plugins
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_DEFAULT_PLUGINS_HPP
#define AI_CORE_DEFAULT_PLUGINS_HPP

namespace ai_core::dnn {

/**
 * @brief Registers all built-in pre/postprocess plugins and the inference
 * engines compiled into this build.
 *
 * Idempotent and thread-safe. The facades (AlgoPreproc / AlgoInferEngine /
 * AlgoPostproc) call it during initialize(), so explicit invocation is only
 * needed when talking to the factories directly before any facade exists.
 * Works for both static and shared linkage — registration no longer relies
 * on static initializers surviving the linker.
 */
void registerDefaultPlugins();

} // namespace ai_core::dnn

#endif
