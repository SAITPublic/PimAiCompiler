/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/attributes.h"

namespace estd { // comes from 'extended std'

template <typename M,
          typename K,
          typename = std::enable_if_t<!std::is_pointer_v<typename std::decay_t<M>::mapped_type>>>
ALWAYS_INLINE auto at_or_null(M&& m, const K& k) -> decltype(&m.find(k)->second) {
    auto it = m.find(k);
    return it == m.end() ? nullptr : &it->second;
}

template <typename M, typename K, typename = std::enable_if_t<std::is_pointer_v<typename std::decay_t<M>::mapped_type>>>
ALWAYS_INLINE auto at_or_null(M&& m, const K& k) -> decltype(m.find(k)->second) {
    auto it = m.find(k);
    return it == m.end() ? nullptr : it->second;
}

template <typename M, typename K>
ALWAYS_INLINE auto at_or_default(M&& m, const K& k, const typename std::decay_t<M>::mapped_type& fallback)
    -> decltype(m.find(k)->second) {
    auto it = m.find(k);
    return it == m.end() ? fallback : it->second;
}

// TODO(someone): remove after we move to C++20
template <typename M, typename Pred>
auto erase_if(M& map, Pred pred) -> decltype(map.erase(map.begin()->first), map.size()) {
    auto old_size = map.size();
    for (auto it = map.begin(), end = map.end(); it != end;) {
        if (pred(*it)) {
            it = map.erase(it);
            continue;
        }
        ++it;
    }
    return old_size - map.size();
}

} // namespace estd
