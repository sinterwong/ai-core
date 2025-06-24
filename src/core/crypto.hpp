#ifndef INFER_ENCRYPT_CRYPTO_HPP
#define INFER_ENCRYPT_CRYPTO_HPP

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace infer::encrypt {

// Forward declarations
namespace detail {
class CryptoImpl;
}

// Encryption configuration
struct CryptoConfig {
  std::string key;
  std::string iv;
  std::string algorithm = "AES"; // Default to AES
  std::string mode = "CBC";      // Default to CBC mode
  size_t keySize = 256;          // Default to AES-256
};

// Main crypto class
class Crypto {
public:
  explicit Crypto(const CryptoConfig &config);
  ~Crypto();

  // File operations
  bool encryptFile(const std::string &inputPath, const std::string &outputPath);
  bool decryptFile(const std::string &inputPath, const std::string &outputPath);

  // Memory operations
  bool encryptData(const std::vector<uint8_t> &input,
                   std::vector<uint8_t> &output);
  bool decryptData(const std::vector<uint8_t> &input,
                   std::vector<uint8_t> &output);

  bool encryptData(const std::string &inputPath,
                   std::vector<uint8_t> &output);

  bool decryptData(const std::string &inputPath,
                   std::vector<uint8_t> &output);

  // Compression operations
  bool compress(const std::string &input, std::string &output);
  bool decompress(const std::string &input, std::string &output);

  // Utility functions
  static std::string
  calculateFileHash(const std::string &filePath,
                    const std::string &hashAlgorithm = "SHA256");
  static CryptoConfig generateSecureConfig();

  static CryptoConfig deriveKeyFromCommit(const std::string &commit);

private:
  std::unique_ptr<detail::CryptoImpl> impl_;
};

// Exception class for crypto operations
class CryptoException : public std::runtime_error {
public:
  explicit CryptoException(const std::string &message)
      : std::runtime_error(message) {}
};

} // namespace infer::encrypt
#endif