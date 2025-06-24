#ifndef INFER_ENCRYPT_CRYPTO_IMPL_HPP
#define INFER_ENCRYPT_CRYPTO_IMPL_HPP

#include "crypto.hpp"
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <memory>

namespace infer::encrypt::detail {

class CryptoImpl {
public:
  explicit CryptoImpl(const CryptoConfig &config);

  bool encryptFile(const std::string &inputPath, const std::string &outputPath);
  bool decryptFile(const std::string &inputPath, const std::string &outputPath);
  bool encryptData(const std::vector<uint8_t> &input,
                   std::vector<uint8_t> &output);
  bool decryptData(const std::vector<uint8_t> &input,
                   std::vector<uint8_t> &output);
  bool encryptData(const std::string &inputPath,
                   std::vector<uint8_t> &output);
  bool decryptData(const std::string &inputPath,
                   std::vector<uint8_t> &output);
  bool compress(const std::string &input, std::string &output);
  bool decompress(const std::string &input, std::string &output);

private:
  CryptoConfig config_;
  std::unique_ptr<CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption> encryptor_;
  std::unique_ptr<CryptoPP::CBC_Mode<CryptoPP::AES>::Decryption> decryptor_;

  void initializeCipher();
};

} // namespace infer::encrypt::detail
#endif