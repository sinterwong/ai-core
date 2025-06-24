#include "crypto_impl.hpp"
#include <cryptopp/files.h>
#include <cryptopp/filters.h>
#include <cryptopp/modes.h>
#include <cryptopp/zdeflate.h>
#include <cryptopp/zinflate.h>

namespace infer::encrypt::detail {

CryptoImpl::CryptoImpl(const CryptoConfig &config) : config_(config) {
  initializeCipher();
}

void CryptoImpl::initializeCipher() {
  // Initialize encryption objects
  encryptor_ =
      std::make_unique<CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption>();
  decryptor_ =
      std::make_unique<CryptoPP::CBC_Mode<CryptoPP::AES>::Decryption>();

  // Set key and IV
  encryptor_->SetKeyWithIV(
      reinterpret_cast<const CryptoPP::byte *>(config_.key.data()),
      config_.key.size(),
      reinterpret_cast<const CryptoPP::byte *>(config_.iv.data()));

  decryptor_->SetKeyWithIV(
      reinterpret_cast<const CryptoPP::byte *>(config_.key.data()),
      config_.key.size(),
      reinterpret_cast<const CryptoPP::byte *>(config_.iv.data()));
}

bool CryptoImpl::encryptFile(const std::string &inputPath,
                             const std::string &outputPath) {
  try {
    CryptoPP::FileSource tmp(
        inputPath.c_str(), true,
        new CryptoPP::StreamTransformationFilter(
            *encryptor_, new CryptoPP::FileSink(outputPath.c_str())));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}

bool CryptoImpl::decryptFile(const std::string &inputPath,
                             const std::string &outputPath) {
  try {
    CryptoPP::FileSource tmp(
        inputPath.c_str(), true,
        new CryptoPP::StreamTransformationFilter(
            *decryptor_, new CryptoPP::FileSink(outputPath.c_str())));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}

bool CryptoImpl::encryptData(const std::vector<uint8_t> &input,
                             std::vector<uint8_t> &output) {
  try {
    output.clear();
    std::vector<CryptoPP::byte> temp(input.begin(), input.end());
    CryptoPP::VectorSource tmp(
        temp, true,
        new CryptoPP::StreamTransformationFilter(
            *encryptor_, new CryptoPP::VectorSink(output)));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}

bool CryptoImpl::decryptData(const std::vector<uint8_t> &input,
                             std::vector<uint8_t> &output) {
  try {
    output.clear();
    std::vector<CryptoPP::byte> temp(input.begin(), input.end());
    CryptoPP::VectorSource tmp(
        temp, true,
        new CryptoPP::StreamTransformationFilter(
            *decryptor_, new CryptoPP::VectorSink(output)));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}

bool CryptoImpl::encryptData(const std::string &inputPath,
                             std::vector<uint8_t> &output) {
  try {
    output.clear();
    CryptoPP::FileSource fileSource(
        inputPath.c_str(), true,
        new CryptoPP::StreamTransformationFilter(
            *encryptor_, new CryptoPP::VectorSink(output)));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}

bool CryptoImpl::decryptData(const std::string &inputPath,
                             std::vector<uint8_t> &output) {
  try {
    output.clear();
    CryptoPP::FileSource fileSource(
        inputPath.c_str(), true,
        new CryptoPP::StreamTransformationFilter(
            *decryptor_, new CryptoPP::VectorSink(output)));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}

bool CryptoImpl::compress(const std::string &input, std::string &output) {
  try {
    output.clear();
    CryptoPP::StringSource tmp(
        input, true, new CryptoPP::Deflator(new CryptoPP::StringSink(output)));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}

bool CryptoImpl::decompress(const std::string &input, std::string &output) {
  try {
    output.clear();
    CryptoPP::StringSource tmp(
        input, true, new CryptoPP::Inflator(new CryptoPP::StringSink(output)));
    return true;
  } catch (const CryptoPP::Exception &) {
    return false;
  }
}
} // namespace infer::encrypt::detail