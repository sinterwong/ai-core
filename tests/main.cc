#include <gtest/gtest.h>
#include <ai_core/plugin_manager.hpp>

int main(int argc, char **argv) {
#ifdef AI_CORE_TEST_PREPROC_PLUGIN
  ai_core::dnn::PluginManager::instance().load(AI_CORE_TEST_PREPROC_PLUGIN);
  ai_core::dnn::PluginManager::instance().load(AI_CORE_TEST_POSTPROC_PLUGIN);
#endif
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
