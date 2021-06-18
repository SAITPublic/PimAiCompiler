/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include <functional>

/*
 * @brief: A very simple implementation of 'Hash Combine' idiom from boost, needed to simplify custom
 *         hash_combine creation. See http://www.boost.org/doc/libs/1_35_0/doc/html/hash/combine.html
 *         for the detailed discussion about the idiom.
 */

template <typename T>
void hashCombine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
void hashVal(std::size_t& seed, const T& val) {
    hashCombine(seed, val);
}

template <typename T, typename... Types>
void hashVal(std::size_t& seed, const T& val, const Types&... args) {
    hashCombine(seed, val);
    hashVal(seed, args...);
}

template <typename... Types>
std::size_t hashVal(const Types&... args) {
    std::size_t seed = 0;
    hashVal(seed, args...);
    return seed;
}

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return hashVal(p.first, p.second);
    }
};

struct TupleHash {
    template <typename... Types>
    std::size_t operator()(const std::tuple<Types...>& t) const {
        return tupleHashImpl(t, std::make_index_sequence<std::tuple_size<std::tuple<Types...>>::value>{});
    }

 private:
    template <typename TTuple, std::size_t... Idx>
    std::size_t tupleHashImpl(const TTuple& t, std::index_sequence<Idx...>) const {
        return hashVal(std::get<Idx>(t)...);
    }
};

struct RangeHash {
    template <typename RangeT>
    std::size_t operator()(const RangeT& range) const {
        std::size_t seed = 0;
        for (const auto& item : range) {
            hashVal(seed, item);
        }
        return seed;
    }
};
