#pragma once

#include "common/include/command_line_parser.hpp"
#include "compiler/include/middlend/common/log.hpp"

#include <unordered_map>

namespace nn_compiler {

/// @brief this class is responsible for holding pass trait that can be shared with other passes
/// @note TraitManager takes ownership for all trait and all kinds of trait have unique instances.
///       This means that if different passes refer to the same trait name (e.g. expansion_ratio),
///       they will refer to the same value (e.g. 2) of expansion_ratio trait in TraitManager
class TraitManager {
 public:
    /// @brief get trait value by name
    template <typename TraitT, typename PassT>
    const TraitT getTrait(const std::string& trait_name) const;

    /// @brief create new pass trait
    void createTrait(const std::string& pass_name, const std::string& trait_name, const std::string& value) {
        auto it = traitname_to_value_.find(trait_name);
        if (it == traitname_to_value_.end()) {
            traitname_to_value_.emplace(trait_name, value);
        } else {
            Log::ME::E_IF(it->second != value)
                << "Pass: `" << pass_name << "` tries to reassign trait: `" << trait_name << "` to another value";
        }
        pass_to_traitname_.emplace(pass_name, trait_name);
    }

 private:
    std::unordered_map<std::string, std::string>      traitname_to_value_;
    std::unordered_multimap<std::string, std::string> pass_to_traitname_;
};

/// @brief get trait value by name
template <typename TraitT, typename PassT>
const TraitT TraitManager::getTrait(const std::string& trait_name) const {
    using decayPassT = std::remove_pointer_t<std::decay_t<PassT>>;

    const auto& pass_name = decayPassT::getName();
    auto        it        = traitname_to_value_.find(trait_name);

    Log::ME::E_IF(it == traitname_to_value_.end()) << "Required pass trait: `" << trait_name.c_str()
                                                   << "` was not assigned for pass: `" << pass_name.c_str() << "`";

    // check that pass can request trait
    auto range               = pass_to_traitname_.equal_range(pass_name);
    bool pass_can_have_trait = std::any_of(range.first, range.second, [&trait_name](const auto& pass_to_traitname) {
        return trait_name == pass_to_traitname.second;
    });

    Log::ME::E_IF(!pass_can_have_trait) << "`" << pass_name.c_str() << "` pass didn't register required trait: `"
                                        << trait_name.c_str() << "`";

    using T = std::decay_t<TraitT>;
    if constexpr (std::is_same_v<T, std::string>) {
        return it->second;
    }
    if constexpr (std::is_same_v<T, double>) {
        return std::stod(it->second);
    }
    if constexpr (std::is_same_v<T, float>) {
        return std::stof(it->second);
    }
    if constexpr (std::is_same_v<T, bool>) {
        return it->second == "true" || it->second == "True";
    }
    if constexpr (std::is_same_v<T, int>) {
        return std::stoi(it->second);
    }
    if constexpr (std::is_same_v<T, long>) {
        return std::stol(it->second);
    }
    if constexpr (std::is_same_v<T, long long>) {
        return std::stoll(it->second);
    }
    if constexpr (std::is_same_v<T, unsigned int>) {
        if (it->second == "-1") {
            return std::numeric_limits<unsigned int>::max();
        }
        unsigned long result = std::stoul(it->second);
        if (result <= std::numeric_limits<unsigned int>::max()) {
            return result;
        }
    }
    if constexpr (std::is_same_v<T, unsigned long>) {
        if (it->second == "-1") {
            return std::numeric_limits<unsigned long>::max();
        }
        return std::stoul(it->second);
    }
    if constexpr (std::is_same_v<T, unsigned long long>) {
        if (it->second == "-1") {
            return std::numeric_limits<unsigned long long>::max();
        }
        return std::stoull(it->second);
    }
    Log::ME::E() << "Cannot convert json value to requested type while parsing trait: `" << trait_name << "` in pass: `"
                 << pass_name << '`';
}

} // namespace nn_compiler
