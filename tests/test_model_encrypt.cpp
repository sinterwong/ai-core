#include "crypto.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>

namespace fs = std::filesystem;

class ModelEncryptTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = fs::path("data/yolov11");
  fs::path modelDir = fs::path("models");

  const std::string commitCode = "bea4f2fe1875e12e5abcb4d40f85d99262ed3054";
};

TEST_F(ModelEncryptTest, EncryptDecrypt) {
#ifdef USE_NCNN
  auto nonEncFile = (modelDir / "yolov11n.ncnn.bin").string();
  auto encFile = "yolov11n.enc.ncnn.bin";
#else
  auto nonEncFile = (modelDir / "yolov11n.onnx").string();
  auto encFile = "yolov11n.enc.onnx";
#endif
  ASSERT_TRUE(fs::exists(nonEncFile));

  auto cryptoConfig = encrypt::Crypto::deriveKeyFromCommit(commitCode);
  encrypt::Crypto crypto(cryptoConfig);
  std::vector<uchar> encData;

  // encrypt file
  crypto.encryptFile(nonEncFile, encFile);

  // decrypt file
  std::vector<uchar> decData;
  crypto.decryptData(encFile, decData);

  // compare
  std::vector<uchar> nonEncData;
  std::ifstream ifs(nonEncFile, std::ios::binary);
  if (!ifs.is_open()) {
    FAIL() << "Failed to open file: " << nonEncFile;
  }
  ifs.seekg(0, std::ios::end);
  nonEncData.resize(ifs.tellg());
  ifs.seekg(0);
  ifs.read((char *)nonEncData.data(), nonEncData.size());
  ASSERT_EQ(decData.size(), nonEncData.size());
  ASSERT_EQ(memcmp(decData.data(), nonEncData.data(), decData.size()), 0);
  fs::remove(encFile);
}
