#include <ai_core/plugin_manager.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

using ai_core::dnn::PluginManager;
using ai_core::dnn::PluginRegistry;

TEST(PluginManager, RejectsVersionMismatchAndRollsBackRegistration) {
  EXPECT_FALSE(PluginRegistry::instance().containsInferenceEngine(
      "RollbackProbe"));
  EXPECT_THROW(PluginManager::instance().load(AI_CORE_TEST_BAD_VERSION_PLUGIN),
               std::runtime_error);
  EXPECT_FALSE(PluginRegistry::instance().containsInferenceEngine(
      "RollbackProbe"));
}

TEST(PluginManager, ExplicitDirectoryDiscoveryUsesPluginNamingConvention) {
  const auto plugins = PluginManager::instance().loadDirectory(
      std::filesystem::path(AI_CORE_TEST_PREPROC_PLUGIN).parent_path());
  EXPECT_TRUE(std::ranges::any_of(plugins, [](const auto &plugin) {
    return plugin.name == "ai_core.preproc.opencv";
  }));
  EXPECT_TRUE(std::ranges::any_of(plugins, [](const auto &plugin) {
    return plugin.name == "ai_core.postproc.opencv";
  }));
}
