/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "common/include/command_line_parser.hpp"
#include "common/include/algorithm_ext.hpp"
#include "common/include/utility.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace cl_opt {

namespace {

Option<std::string> help("--help", "", "print available options");
Option<std::string> help_hidden("--help-hidden", "", "print available options including hidden");

bool isValueValid(const BaseOption* opt, const char* val) {
    const auto& valid_vals = opt->getValidValues();

    if (valid_vals.empty()) {
        return true;
    }

    return estd::contains(valid_vals, val);
}

} // anonymous namespace

void CommandLineParser::checkRequiredOptions(const std::unordered_set<std::string>& cmd_opts) const {
    for (const auto& opt : options_) {
        if (!opt->isRequired()) {
            continue;
        }

        bool found =
            estd::any_of(opt->getNames(), [&cmd_opts](const std::string& name) { return cmd_opts.count(name); });

        if (!found) {
            usage("Required argument: `" + opt->getNames().front() + "` must be passed from command line");
        }
    }
}

void CommandLineParser::usage(const std::string& msg, int exit_code, bool print_hidden) const {
    if (!msg.empty()) {
        std::cerr << msg << "\n";
    }

    std::cerr << "Usage: " << prog_name_ << " ARGUMENTS\n";
    std::cerr << "Available ARGUMENTS" << std::endl;

    auto max_len = getMaxOptNamesLen(print_hidden);
    for (const auto* opt : options_) {
        if (!print_hidden && opt->isHidden()) {
            // don't consider hidden options
            continue;
        }

        const auto& descr      = opt->getDescription();
        const auto& names      = opt->getNames();
        const auto& value_hint = opt->getValueHint();

        std::string all_names = estd::strJoin(names, ", ");
        if (!value_hint.empty()) {
            all_names += " " + value_hint;
        }

        std::string spaces(max_len - all_names.length(), ' ');
        std::cerr << "        " << all_names << spaces << " -    " << descr << std::endl;
    }

    exit(exit_code);
}

std::string::size_type CommandLineParser::getMaxOptNamesLen(bool print_hidden) const {
    using size_type = std::string::size_type;

    size_type max_len = 0;

    for (const auto* opt : options_) {
        if (!print_hidden && opt->isHidden()) {
            // don't consider hidden options
            continue;
        }

        const auto&        names      = opt->getNames();
        const std::string& value_hint = opt->getValueHint();

        size_type value_hint_len = value_hint.empty() ? 0 : value_hint.length() + 1;
        size_type separators_len = 2 * (names.size() - 1);
        size_type names_len      = 0;

        names_len = estd::accumulate(
            names, names_len, [](size_type sum, const std::string& str) { return sum + str.length(); });
        max_len = std::max(max_len, names_len + separators_len + value_hint_len);
    }

    return max_len;
}

void CommandLineParser::parseCommandLine(int argc, char** argv) {
    assert(!options_.empty());

    prog_name_ = argv[0];
    std::unordered_set<std::string> cmd_opts;
    cmd_opts.reserve(argc);

    // first scan to find help options
    for (int i = 1; i < argc; i++) {
        if (argv[i] == help.getName()) {
            usage("", EXIT_SUCCESS);
        }

        if (argv[i] == help_hidden.getName()) {
            usage("", EXIT_SUCCESS, true);
        }
    }

    for (int i = 1; i < argc; i++) {
        auto it = names_to_opt_.find(argv[i]);
        if (argv[i][0] != '-' || it == names_to_opt_.end()) {
            usage("Invalid argument `" + std::string(argv[i]) + "`");
        }

        BaseOption* opt = it->second;
        options_.insert(opt);
        cmd_opts.emplace(argv[i]);

        // if option can have no value and the next command line argument
        // starts with `-` or doesn't exist then suggest that it's OK
        if (!opt->mustHaveValue() && ((i + 1) == argc || argv[i + 1][0] == '-')) {
            // But we need to set default value in this case anyway
            opt->setValue(std::string());
            continue;
        }

        // if option must have value but arguments are over then report error
        if ((i + 1) == argc) {
            usage("This argument `" + std::string(argv[i]) + "` must have value");
        }

        // if value can be as option name and option must have value then suggest that value is missed
        if (names_to_opt_.find(argv[i + 1]) != names_to_opt_.end()) {
            usage("Missed value for argument `" + std::string(argv[i]) + "`");
        }

        if (!isValueValid(opt, argv[i + 1])) {
            auto        vec_vals = opt->getValidValues();
            std::string vals;
            for (const auto& v : vec_vals) {
                vals.append(v + ", ");
            }
            if (!vec_vals.empty()) {
                // remove last ", " symbols
                vals.pop_back();
                vals.pop_back();
            }

            usage("Invalid value for argument `" + std::string(argv[i]) + "`. Valid values are `" + vals + "`");
        }

        i++; // skip value for argument
        opt->setValue(argv[i]);
    }

    checkRequiredOptions(cmd_opts);
}

} // namespace cl_opt
