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
#include <sstream>
#include <utility>

namespace estd {

namespace details {

struct abstract_empty {
    virtual ~abstract_empty() = 0;
};

} // namespace details

/// @brief check that type is empty. It also considers abstract empty classes as empty (std::is_empty_v doesn't do that)
template <typename T>
inline constexpr bool is_empty = std::is_empty_v<T> ||
                                 (std::is_polymorphic_v<T> && sizeof(T) == sizeof(details::abstract_empty));

/**
 * @brief   Joins the strings acquired from elements of range [first, last) using given separator between elements
 * @details Strings acquired from elements using stream insertion operator
 * @returns concatenated string
 */
template <typename InputIteratorT>
std::string strJoin(InputIteratorT first, InputIteratorT last, const std::string& separator = "") {
    assert(first != last);

    std::stringstream ss;

    ss << *first;
    for (auto iter = ++first; iter != last; ++iter) {
        ss << separator << *iter;
    }

    return ss.str();
}

/**
 * @brief   Joins the strings acquired from elements of range using given separator between elements
 * @details Strings acquired from elements using stream insertion operator
 * @returns concatenated string
 */
template <typename RangeT>
std::string strJoin(RangeT&& range, const std::string& separator = "") {
    return strJoin(std::begin(range), std::end(range), separator);
}

} // namespace estd
