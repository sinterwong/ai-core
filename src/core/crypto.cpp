#include "crypto.hpp"
#include "crypto_impl.hpp"
#include <cryptopp/files.h>
#include <cryptopp/filters.h>
#include <cryptopp/hex.h>
#include <cryptopp/osrng.h>
#include <cryptopp/sha.h>
#include <cryptopp/zdeflate.h>
#include <cryptopp/zinflate.h>

namespace infer::encrypt {

Crypto::Crypto(const CryptoConfig &config)
    : impl_(std::make_unique<detail::CryptoImpl>(config)) {}

Crypto::~Crypto() = default;

bool Crypto::encryptFile(const std::string &inputPath,
                         const std::string &outputPath) {
  try {
    return impl_->encryptFile(inputPath, outputPath);
  } catch (const CryptoPP::Exception &e) {
    throw CryptoException("Encryption failed: " + std::string(e.what()));
  }
}

bool Crypto::decryptFile(const std::string &inputPath,
                         const std::string &outputPath) {
  try {
    return impl_->decryptFile(inputPath, outputPath);
  } catch (const CryptoPP::Exception &e) {
    throw CryptoException("Decryption failed: " + std::string(e.what()));
  }
}

bool Crypto::encryptData(const std::vector<uint8_t> &input,
                         std::vector<uint8_t> &output) {
  try {
    return impl_->encryptData(input, output);
  } catch (const CryptoPP::Exception &e) {
    throw CryptoException("Data encryption failed: " + std::string(e.what()));
  }
}

bool Crypto::decryptData(const std::vector<uint8_t> &input,
                         std::vector<uint8_t> &output) {
  try {
    return impl_->decryptData(input, output);
  } catch (const CryptoPP::Exception &e) {
    throw CryptoException("Data decryption failed: " + std::string(e.what()));
  }
}

bool Crypto::encryptData(const std::string &inputPath,
                         std::vector<uint8_t> &output) {
  try {
    return impl_->encryptData(inputPath, output);
  } catch (const CryptoPP::Exception &e) {
    throw CryptoException("Data encryption failed: " + std::string(e.what()));
  }
}

bool Crypto::decryptData(const std::string &inputPath,
                         std::vector<uint8_t> &output) {
  try {
    return impl_->decryptData(inputPath, output);
  } catch (const CryptoPP::Exception &e) {
    throw CryptoException("Data decryption failed: " + std::string(e.what()));
  }
}

bool Crypto::compress(const std::string &input, std::string &output) {
  return impl_->compress(input, output);
}

bool Crypto::decompress(const std::string &input, std::string &output) {
  return impl_->decompress(input, output);
}

std::string Crypto::calculateFileHash(const std::string &filePath,
                                      const std::string &hashAlgorithm) {
  try {
    std::string result;
    if (hashAlgorithm == "SHA256") {
      CryptoPP::SHA256 hash;
      CryptoPP::FileSource(filePath.c_str(), true,
                           new CryptoPP::HashFilter(
                               hash, new CryptoPP::HexEncoder(
                                         new CryptoPP::StringSink(result))));
    } else {
      throw CryptoException("Unsupported hash algorithm: " + hashAlgorithm);
    }
    return result;
  } catch (const CryptoPP::Exception &e) {
    throw CryptoException("Hash calculation failed: " + std::string(e.what()));
  }
}

CryptoConfig Crypto::generateSecureConfig() {
  CryptoConfig config;
  CryptoPP::AutoSeededRandomPool rng;

  // Generate random key
  std::vector<CryptoPP::byte> key(32); // 256 bits
  rng.GenerateBlock(key.data(), key.size());
  config.key.assign(reinterpret_cast<char *>(key.data()), key.size());

  // Generate random IV
  std::vector<CryptoPP::byte> iv(16); // 128 bits
  rng.GenerateBlock(iv.data(), iv.size());
  config.iv.assign(reinterpret_cast<char *>(iv.data()), iv.size());

  return config;
}

CryptoConfig Crypto::deriveKeyFromCommit(const std::string &commit) {
  CryptoConfig config;
  CryptoPP::SHA256 hash;
  std::string derivedKey;

  CryptoPP::StringSource(
      commit, true,
      new CryptoPP::HashFilter(hash, new CryptoPP::StringSink(config.key)));

  std::string ivInput = commit + "iv_salt";
  CryptoPP::StringSource(
      ivInput, true,
      new CryptoPP::HashFilter(hash, new CryptoPP::StringSink(derivedKey)));
  config.iv = derivedKey.substr(0, 16); // Take first 16 bytes for IV
  return config;
}

} // namespace infer::encrypt