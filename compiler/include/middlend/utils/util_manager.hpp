#pragma once

#include "compiler/include/common/log.hpp"

#include "common/include/ctti.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace nn_compiler {

/// @brief internal classes to support implementation of concept based polymorphism
namespace internal {

/// @brief interface for utils
struct UtilConcept {
    virtual ~UtilConcept() = default;
};

/// @brief This class is container for utils
template <typename UtilT>
struct UtilModel final : UtilConcept {
    explicit UtilModel(std::unique_ptr<UtilT> util) : util(std::move(util)) {}

    std::unique_ptr<UtilT> util;
};

} // namespace internal

/// @brief This is util manager class that registers utils that will be used by passes via PassManager
///        Implementation is based on concepts-based polymorphism that was introduced by Sean Parent.
///        See description of this idiom:
///        https://sean-parent.stlab.cc/papers-and-presentations/#value-semantics-and-concept-based-polymorphism
class UtilManager {
 public:
    UtilManager()                       = default;
    UtilManager(UtilManager&&) noexcept = delete;

    /// @brief get util with type UtilT that was registered before
    template <typename UtilT, typename PassT>
    const UtilT* getUtil() const {
        return getUtil<UtilT, PassT>(true);
    }

    /// @brief get util with type UtilT that was registered before
    template <typename UtilT, typename PassT>
    const UtilT& getUtilRef() const {
        return *getUtil<UtilT, PassT>(true);
    }

    /// @brief register util for pass
    template <typename UtilT>
    void addUtil(UtilT util, const std::string& pass_name) {
        addUtil(std::move(util), ctti::typeId<UtilT>(), pass_name);
    }

    /// @brief register util for pass, but allow pass to request util using base class type
    template <typename BaseUtilT, typename UtilT>
    void addUtilAsBase(UtilT util, const std::string& pass_name) {
        static_assert(std::is_base_of_v<BaseUtilT, UtilT>, "invalid base util");
        addUtil(std::move(util), ctti::typeId<BaseUtilT>(), pass_name);
    }

 private:
    /// @brief A queue that preserves its elements throughout the compilation pipeline.
    /// @details we can't use std::queue because we need to preserve all elements.
    ///          All utils must stay alive through the whole compilation pipeline.
    class UtilQueue {
     public:
        void push(std::unique_ptr<internal::UtilConcept> elem) { queue_.push_back(std::move(elem)); }

        auto& pop() const {
            LOGE_IF(ME, queue_.size() <= cur_idx_, "util queue overflow");
            auto& util = queue_[cur_idx_];
            ++cur_idx_;
            return util;
        }

     private:
        std::vector<std::unique_ptr<internal::UtilConcept>> queue_;
        mutable size_t                                      cur_idx_ = 0;
    };

    template <typename UtilT>
    void addUtil(UtilT util, ctti::TypeIndex util_id, const std::string& pass_name) {
        auto key = ctti::TypeIndex(util_id);
        auto val = std::make_unique<internal::UtilModel<UtilT>>(std::make_unique<UtilT>(std::move(util)));

        utils_[key].push(std::move(val));
        pass_to_util_.emplace(pass_name, key);
    }

    template <typename UtilT, typename PassT>
    const UtilT* getUtil(bool fail_if_not_present) const {
        auto util_id = ctti::TypeIndex(ctti::typeId<UtilT>());

        auto it = utils_.find(util_id);

        if (it == utils_.end()) {
            LOGE_IF(ME, fail_if_not_present, "Required util was not saved\n");
            return nullptr;
        }

        auto& util_queue = it->second;

        auto& util_concept = util_queue.pop();
        auto* util_model   = static_cast<internal::UtilModel<UtilT>*>(util_concept.get());

        // check that pass can request util
        using decayPassT = std::remove_pointer_t<std::decay_t<PassT>>;
        auto pass_name   = decayPassT::getName();
        auto range       = pass_to_util_.equal_range(pass_name);

        bool pass_can_have_util = std::any_of(
            range.first, range.second, [util_id](const auto& pass_to_util) { return util_id == pass_to_util.second; });

        LOGE_IF(ME, !pass_can_have_util, "`%s` pass didn't register required util", pass_name.c_str());

        LOGE_IF(ME, !util_model->util, "invalid util");
        return util_model->util.get();
    }

    std::unordered_map<ctti::TypeIndex, UtilQueue>        utils_;
    std::unordered_multimap<std::string, ctti::TypeIndex> pass_to_util_;
};

} // namespace nn_compiler
