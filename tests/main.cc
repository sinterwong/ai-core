#include <ai_core/plugin_manager.hpp>
#include <gtest/gtest.h>

#include "plugin_test_path.hpp"

const char *ai_core_test_argv0 = nullptr;

int main(int argc, char **argv) {
  ai_core_test_argv0 = argv[0];
#ifdef AI_CORE_TEST_PREPROC_PLUGIN
  ai_core::dnn::PluginManager::instance().load(
      aiCoreTestPluginPath(argv[0], AI_CORE_TEST_PREPROC_PLUGIN).string());
  ai_core::dnn::PluginManager::instance().load(
      aiCoreTestPluginPath(argv[0], AI_CORE_TEST_POSTPROC_PLUGIN).string());
#endif
#ifdef AI_CORE_TEST_ORT_PLUGIN
  ai_core::dnn::PluginManager::instance().load(
      aiCoreTestPluginPath(argv[0], AI_CORE_TEST_ORT_PLUGIN).string());
#endif
#ifdef AI_CORE_TEST_NCNN_PLUGIN
  ai_core::dnn::PluginManager::instance().load(
      aiCoreTestPluginPath(argv[0], AI_CORE_TEST_NCNN_PLUGIN).string());
#endif
#ifdef AI_CORE_TEST_TRT_PLUGIN
  ai_core::dnn::PluginManager::instance().load(
      aiCoreTestPluginPath(argv[0], AI_CORE_TEST_TRT_PLUGIN).string());
#endif
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
