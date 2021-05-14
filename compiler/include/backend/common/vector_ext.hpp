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

#include <vector>

namespace estd { // comes from 'extended std'

// TODO(someone): remove after we move to C++20
template <typename V, typename A, typename Pred>
auto erase(std::vector<V, A>& v, const V& val) {
    auto it         = std::remove(v.begin(), v.end(), val);
    auto num_erased = std::distance(it, v.end());
    v.erase(it, v.end());
    return num_erased;
}

// TODO(someone): remove after we move to C++20
template <typename V, typename A, typename Pred>
auto erase_if(std::vector<V, A>& v, Pred pred) {
    auto it         = std::remove_if(v.begin(), v.end(), pred);
    auto num_erased = std::distance(it, v.end());
    v.erase(it, v.end());
    return num_erased;
}

} // namespace estd
