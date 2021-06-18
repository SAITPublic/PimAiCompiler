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

#include <cassert>
#include <cstring>
#include <string>

#include "common/include/iterator_ext.hpp"
#include "common/include/log.hpp"

/// @brief function that returns typename FuncT in string representation
template <typename FuncT>
inline std::string getTypeName() {
    std::string full_name = __PRETTY_FUNCTION__;

    // format of string for this function that represented by PRETTY_FUNCTION is similar to
    //
    //   (gcc)   std::__cxx11::string getTypeName() [with FuncT = InstantiatedType;
    //           std::__cxx11::string = std::__cxx11::basic_string<char>]
    //   (clang) std::__cxx11::string getTypeName() [FuncT = InstantiatedType]
    //
    // We just need to remove all from string except InstantiatedType
    const char template_name[] = "FuncT = ";

    auto type_name = full_name.substr(full_name.find(template_name));
    type_name.erase(0, sizeof(template_name) - 1);

    auto last_char_pos = type_name.find(';');
    if (last_char_pos == std::string::npos) {
        last_char_pos = type_name.find(']');
    }
    type_name.erase(last_char_pos);

    // remove namespace if it presents
    auto ns_pos = type_name.find("::");
    if (ns_pos != std::string::npos) {
        type_name.erase(0, ns_pos + 2);
    }

    return type_name;
}

template <typename Iterable>
size_t cap_distance(Iterable from, Iterable to, size_t cap) {
    size_t d = 0;
    while (from != to && d <= cap) {
        ++from;
        ++d;
    }
    return d;
}

/// @brief wrapper that allows to print arbitrary containers/ranges
///        It can be used as follows: `LOG(ME::D()) << PrintRange(obj);`
template <typename RangeT>
struct PrintRange {
    explicit PrintRange(const RangeT& range, const char* indent = ", ") : range(range), indent(indent) {}

    explicit PrintRange(const RangeT& range, size_t max_length, const char* indent = ", ")
        : range(range), max_length(max_length), indent(indent) {}

    friend std::ostream& operator<<(std::ostream& s, const PrintRange<RangeT>& pr) {
        const bool   trim          = cap_distance(pr.range.begin(), pr.range.end(), pr.max_length + 1) > pr.max_length;
        const size_t prefix_length = trim ? pr.max_length - 1 : pr.max_length;

        size_t printed = 0;
        for (auto it = pr.range.begin(); it != pr.range.end(); ++printed) {
            if (printed >= prefix_length && pr.max_length > 0) {
                break;
            }

            s << *it;
            if (++it != pr.range.end()) {
                s << pr.indent;
            }
        }

        if (trim && pr.max_length > 0) {
            auto last = estd::last(pr.range);
            s << "...";
            if (last != pr.range.end()) {
                s << pr.indent << *last;
            }
        }
        return s;
    }

    const RangeT& range;
    size_t        max_length = 0; // Print all if 0, or print limited number otherwise
    const char*   indent;
};
