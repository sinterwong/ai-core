#ifndef AI_CORE_TYPE_SAFE_FACTORY_HPP
#define AI_CORE_TYPE_SAFE_FACTORY_HPP

#include "ai_core/data_packet.hpp"
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace ai_core {

/**
 * @brief Singleton string->creator registry for plugins.
 *
 * @par Thread safety
 * @ref registerCreator is not synchronized and is expected to run at startup
 * (via an explicit plugin entrypoint) before any concurrent
 * use. Once registration is complete, @ref create and @ref isRegistered are
 * read-only and safe to call concurrently. Do not register from one thread
 * while another creates.
 */
template <class BaseClass> class Factory {
public:
  /** Creator invoked with type-erased construction parameters. */
  using Creator = std::function<std::shared_ptr<BaseClass>(const DataPacket &)>;

  static Factory &instance() {
    static Factory instance;
    return instance;
  }

  /**
   * @brief Add a non-null creator under a unique name.
   * @throws std::runtime_error If `creator` is empty.
   */
  bool registerCreator(const std::string &class_name, Creator creator) {
    if (!creator) {
      throw std::runtime_error("Cannot register a null creator");
    }

    auto [it, success] =
        m_creatorRegistry.insert({class_name, std::move(creator)});
    return success;
  }

  /**
   * @brief Construct the registered derived type.
   * @throws std::runtime_error If the name is absent or construction fails.
   */
  std::shared_ptr<BaseClass> create(const std::string &class_name,
                                    const DataPacket &params = {}) const {
    auto it = m_creatorRegistry.find(class_name);
    if (it == m_creatorRegistry.end()) {
      throw std::runtime_error("Factory error: Class '" + class_name +
                               "' not registered for base type '" +
                               typeid(BaseClass).name() + "'.");
    }

    try {
      return it->second(params);
    } catch (const std::exception &e) {
      throw std::runtime_error("Factory error: Failed to create '" +
                               class_name + "': " + e.what());
    }
  }

  bool isRegistered(const std::string &class_name) const {
    return m_creatorRegistry.count(class_name);
  }

private:
  Factory() = default;
  ~Factory() = default;

  Factory(const Factory &) = delete;
  Factory &operator=(const Factory &) = delete;
  Factory(Factory &&) = delete;
  Factory &operator=(Factory &&) = delete;

  std::map<std::string, Creator> m_creatorRegistry;
};

} // namespace ai_core

#endif
