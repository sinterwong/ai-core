#include <ai_core/plugin_manager.hpp>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
#ifdef AI_CORE_TEST_PREPROC_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_TEST_PREPROC_PLUGIN);
  ai_core::dnn::PluginManager::instance().load(AI_CORE_TEST_POSTPROC_PLUGIN);
#endif
#ifdef AI_CORE_TEST_ORT_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_TEST_ORT_PLUGIN);
#endif
#ifdef AI_CORE_TEST_NCNN_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_TEST_NCNN_PLUGIN);
#endif
#ifdef AI_CORE_TEST_TRT_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_TEST_TRT_PLUGIN);
#endif
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
