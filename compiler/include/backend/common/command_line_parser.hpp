/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/log.hpp"
#include "common/types.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cl_opt {

enum class Hidden { YES, NO };
enum class Required { YES, NO };

/// @brief interface for options that is used by command line parser
class BaseOption {
 public:
    explicit BaseOption(std::vector<std::string> names,
                        std::string              val_hint,
                        std::string              descr,
                        bool                     is_required,
                        bool                     is_hidden,
                        std::vector<std::string> valid_vals)
        : names_(std::move(names)), val_hint_(std::move(val_hint)), descr_(std::move(descr)), required_(is_required),
          hidden_(is_hidden), valid_vals_(std::move(valid_vals)) {}

    const std::string&              getName() const { return names_.front(); }
    const std::vector<std::string>& getNames() const { return names_; }
    const std::string&              getDescription() const { return descr_; }
    bool                            isRequired() const { return required_; }
    bool                            isHidden() const { return hidden_; }
    const std::vector<std::string>& getValidValues() const { return valid_vals_; }
    const std::string&              getValueHint() const { return val_hint_; }

    virtual bool hasValue() const                 = 0;
    virtual void setValue(const std::string& val) = 0;
    virtual bool mustHaveValue() const            = 0;

 private:
    std::vector<std::string> names_;
    std::string              val_hint_;
    std::string              descr_;
    bool                     required_;
    bool                     hidden_;
    std::vector<std::string> valid_vals_;
};

/// @brief class for command line parsing and checking correctness of options
class CommandLineParser {
 public:
    /// @brief parse command line
    void parseCommandLine(int argc, char** argv);

    static CommandLineParser& getInstance() {
        static CommandLineParser parser;
        return parser;
    }

    /// @brief clone parser object (will be useful for unit testing)
    std::unique_ptr<CommandLineParser> clone() const {
        std::unique_ptr<CommandLineParser> cloned_parser(new CommandLineParser(*this));
        return cloned_parser;
    }

 private:
    CommandLineParser()                         = default;
    CommandLineParser(const CommandLineParser&) = default;
    CommandLineParser& operator=(const CommandLineParser&) = default;

    /// @brief print usage and exit
    NO_RETURN void usage(const std::string& msg, int exit_code = EXIT_FAILURE, bool print_hidden = false) const;

    std::string::size_type getMaxOptNamesLen(bool print_hidden) const;

    /// @brief check that all required options are present in command line
    void checkRequiredOptions(const std::unordered_set<std::string>& cmd_opts) const;

    template <typename T>
    friend class Option;

    /// @brief register option (observer pattern)
    void registerOption(BaseOption* option) {
        options_.insert(option);

        const auto& names = option->getNames();
        for (const auto& name : names) {
            assert(names_to_opt_.count(name) == 0);
            names_to_opt_[name] = option;
        }
    }

    void unregisterOption(BaseOption* option) {
        options_.erase(option);

        for (const auto& name : option->getNames()) {
            names_to_opt_.erase(name);
        }
    }

    std::unordered_set<BaseOption*>              options_;
    std::unordered_map<std::string, BaseOption*> names_to_opt_;
    std::string                                  prog_name_;
};

/// @brief represents command line option with type TOpt
template <typename TOpt>
class Option : public BaseOption {
 public:
    /// @param name - name of option
    /// @param val_hint - specifies the hint for value in help message
    /// @param descr - textual description of option
    /// @param is_required - if true then option must be present in command line
    /// @param default_val - default value of option
    /// @param valid_vals - if present then option can take only these values
    explicit Option(const std::string&              name,
                    const std::string&              val_hint,
                    const std::string&              descr,
                    Required                        required    = Required::NO,
                    Hidden                          hidden      = Hidden::NO,
                    std::optional<TOpt>             default_val = std::nullopt,
                    const std::vector<std::string>& valid_vals  = std::vector<std::string>())
        : BaseOption({name}, val_hint, descr, required == Required::YES, hidden == Hidden::YES, valid_vals),
          option_(std::move(default_val)) {
        CommandLineParser::getInstance().registerOption(this);
    }

    /// @param name - names of option (e.g. -h, --help)
    /// @param val_hint - specifies the hint for value in help message
    /// @param descr - textual description of option
    /// @param is_required - if true then option must be present in command line
    /// @param default_val - default value of option
    /// @param valid_vals - if present then option can take only these values
    explicit Option(const std::vector<std::string>& names,
                    const std::string&              val_hint,
                    const std::string&              descr,
                    Required                        required    = Required::NO,
                    Hidden                          hidden      = Hidden::NO,
                    std::optional<TOpt>             default_val = std::nullopt,
                    const std::vector<std::string>& valid_vals  = std::vector<std::string>())
        : BaseOption(names, val_hint, descr, required == Required::YES, hidden == Hidden::YES, valid_vals),
          option_(std::move(default_val)) {
        CommandLineParser::getInstance().registerOption(this);
    }

    ~Option() { CommandLineParser::getInstance().unregisterOption(this); }

    /// @brief getters for contained option
    const TOpt& get() const noexcept {
        Log::COMMON::E_IF(!option_.has_value()) << "trying to get undefined option " << getNames().front();
        return *option_;
    }
    TOpt& get() noexcept { return const_cast<TOpt&>(static_cast<const Option*>(this)->get()); }

    operator TOpt() const {
        Log::COMMON::E_IF(!option_.has_value()) << "trying to get undefined option " << getNames().front();
        return *option_;
    }

    // using another conversion operator for non implicitly convertible types (via SFINAE)
    // (maybe from int to enum class). If static_cast is not allowed from TOpt to T
    // we get compile error
    template <typename T, std::enable_if_t<!std::is_convertible_v<T, TOpt>, T*> = nullptr>
    operator T() const {
        Log::COMMON::E_IF(!option_.has_value()) << "trying to get undefined option " << getNames().front();
        return static_cast<T>(*option_);
    }

    // to prevent heap allocation for options
    void* operator new(std::size_t) = delete;
    void  operator delete(void*)    = delete;

    /// @brief implementation of BaseOption interface. Needed fo command line parser
    bool hasValue() const override { return option_.has_value(); }
    void setValue(const std::string& val) override { return static_cast<TOpt>(val); }
    bool mustHaveValue() const override { return true; }

 private:
    std::optional<TOpt> option_;
};

template <>
inline void Option<int>::setValue(const std::string& val) {
    option_ = std::stoi(val);
}

template <>
inline void Option<unsigned>::setValue(const std::string& val) {
    option_ = std::strtoul(val.c_str(), 0, 0);
}

template <>
inline void Option<double>::setValue(const std::string& val) {
    option_ = std::strtod(val.c_str(), 0);
}

template <>
inline void Option<bool>::setValue(const std::string& val) {
    option_ = !(val == "false" || val == "False" || val == "FALSE" || val == "off" || val == "0");
}

template <>
inline void Option<std::string>::setValue(const std::string& val) {
    option_ = val;
}

template <>
inline bool Option<bool>::mustHaveValue() const {
    return false;
}
} // namespace cl_opt
