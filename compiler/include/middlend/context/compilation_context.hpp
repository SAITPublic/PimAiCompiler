#pragma once

#include "common/include/ctti.hpp"
#include "common/include/pretty_print.hpp"

#include "compiler/include/common/log.hpp"

#include <unordered_map>

namespace nn_compiler {

enum class CtxDataMode { INVALID, READ, MODIFY, WRITE, READ_OPTIONAL };

enum class CtxDataScope { LOCAL, GLOBAL };

inline CtxDataMode strToCtxDataMode(const std::string& mode) {
    if (mode == "read")
        return CtxDataMode::READ;
    else if (mode == "modify")
        return CtxDataMode::MODIFY;
    else if (mode == "write")
        return CtxDataMode::WRITE;
    else if (mode == "read_optional")
        return CtxDataMode::READ_OPTIONAL;
    else
        Log::ME::E() << "Invalid Pass Data mode";
}

inline std::string ctxDataModeToStr(const CtxDataMode mode) {
    switch (mode) {
        case CtxDataMode::READ:
            return "read";
        case CtxDataMode::WRITE:
            return "write";
        case CtxDataMode::MODIFY:
            return "modify";
        case CtxDataMode::READ_OPTIONAL:
            return "read_optional";
        default:
            Log::ME::E() << "Invalid CtxData mode";
    }
}

inline std::ostream& operator<<(std::ostream& os, CtxDataMode mode) { return os << ctxDataModeToStr(mode); }

inline CtxDataScope strToCtxDataScope(const std::string& scope) {
    if (scope == "global")
        return CtxDataScope::GLOBAL;
    else
        // set Local for default
        return CtxDataScope::LOCAL;
}

namespace detail {
/// @brief support class to implement type-erasure idiom
struct IPassData {
    virtual ~IPassData() = default;

    virtual std::unique_ptr<IPassData> create() const = 0;
};

/// @brief support class to implement type-erasure idiom
template <typename DataT>
struct PassData final : IPassData {
    std::unique_ptr<IPassData> create() const override { return std::make_unique<PassData>(); }

    DataT data;
};
} // namespace detail

/// @brief this class is responsible for holding pass data that can be shared with other passes
/// @note CompilationContext takes ownership for all data and all kinds of data have unique instances.
///       This means that if different passes refer to the same data type (e.g. VRegionInfo),
///       they will refer to the same instance of VRegionInfo data in CompilationContext
class CompilationContext {
 public:
    /// @brief reset local data and keep global data
    void resetLocalData() {
        // resets local data and call default ctor for it
        for (auto& id_to_info : pass_data_) {
            auto& info = id_to_info.second;

            if (info.data_scope == CtxDataScope::LOCAL) {
                info.data = info.data->create();
            }
        }
    }

    /// @brief get pass data
    template <typename DataT, typename PassT>
    DataT* getData() {
        return getData<DataT, PassT>(true);
    }

    /// @brief get pass data
    template <typename DataT, typename PassT>
    DataT& getDataRef() {
        return *getData<DataT, PassT>(true);
    }

    /// @brief create new pass data
    template <typename DataT>
    void createData(const std::string& pass_name, CtxDataMode mode, CtxDataScope scope) {
        auto ctx_id = ctti::TypeIndex(ctti::typeId<DataT>());

        if (!pass_data_.count(ctx_id)) {
            pass_data_.emplace(ctx_id, CtxInfo(std::make_unique<detail::PassData<DataT>>(), scope));
        }

        pass_to_data_.emplace(pass_name, std::make_pair(ctx_id, mode));
    }

 private:
    template <typename DataT, typename PassT>
    DataT* getData(bool fail_if_not_present) {
        using decayPassT = std::remove_pointer_t<std::decay_t<PassT>>;

        const auto& pass_name = decayPassT::getName();
        const auto& data_name = getTypeName<DataT>();

        auto data_id = ctti::TypeIndex(ctti::typeId<DataT>());
        auto it      = pass_data_.find(data_id);

        if (it == pass_data_.end()) {
            Log::ME::E_IF(fail_if_not_present)
                << "Required pass data: `" << data_name << "` was not saved for pass: `" << pass_name << '`';
            return nullptr;
        }

        auto& ctx_info = it->second;
        auto& pass_res = static_cast<detail::PassData<DataT>&>(*ctx_info.data);

        // check that pass can request data
        auto range = pass_to_data_.equal_range(pass_name);

        bool pass_can_have_data            = false;
        bool pass_with_correct_data_access = false;

        for (auto range_it = range.first; range_it != range.second; ++range_it) {
            auto [ctx_id, ctx_mode] = range_it->second;

            if (data_id == ctx_id) {
                // READ_OPTIONAL allows to use pass data w/o previous initialization.
                // In this case all returned datas have their default values.
                ctx_info.is_valid |= (ctx_mode == CtxDataMode::WRITE || ctx_mode == CtxDataMode::READ_OPTIONAL);

                pass_can_have_data = true;

                // pass that registered data with "read" mode must request it only as `const` value
                pass_with_correct_data_access = (ctx_mode != CtxDataMode::READ || std::is_const_v<DataT>);

                NN_DEBUG(Log::ME::D() << "Pass: `" << pass_name << "` requires data: `" << data_name
                                      << "` with ctx data mode: " << ctx_mode);
                break;
            }
        }

        LOGE_IF(ME,
                !pass_can_have_data,
                "`%s` pass didn't register required data: `%s`",
                pass_name.c_str(),
                data_name.c_str());
        LOGE_IF(ME,
                !pass_with_correct_data_access,
                "`%s` pass requires write access to data marked as read-only: `%s` (may be missing `const` qualifier "
                "in CompilationContext::getData call)",
                pass_name.c_str(),
                data_name.c_str());
        LOGE_IF(ME,
                !ctx_info.is_valid,
                "`%s` pass tries to use invalid data: `%s`\n",
                pass_name.c_str(),
                data_name.c_str());

        return &pass_res.data;
    }

    // holds information for compilation context data
    struct CtxInfo {
        explicit CtxInfo(std::unique_ptr<detail::IPassData> data, CtxDataScope data_scope, bool is_valid = false)
            : data(std::move(data)), data_scope(data_scope), is_valid(is_valid) {}

        std::unique_ptr<detail::IPassData> data;       // data itself
        CtxDataScope                       data_scope; // scope: necessary for multi-core
        bool                               is_valid;   // valid/invalid state: data that wasn't set can't be used
    };

    std::unordered_map<ctti::TypeIndex, CtxInfo>                                  pass_data_;
    std::unordered_multimap<std::string, std::pair<ctti::TypeIndex, CtxDataMode>> pass_to_data_;
};
} // namespace nn_compiler
