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

#include <iterator>

namespace estd { // comes from 'extended std'

/// @brief return iterator on the last element of the container or end iterator if container is empty
/// @note this function takes O(n) time for iterators with std::forward_iterator_tag
template <typename ContainerT>
auto last(ContainerT&& container) {
    using ItType     = std::conditional_t<std::is_const_v<std::remove_reference_t<ContainerT>>,
                                      decltype(std::cend(container)),
                                      decltype(std::end(container))>;
    using ItCategory = typename std::iterator_traits<ItType>::iterator_category;

    ItType beg = std::begin(container);
    ItType end = std::end(container);

    if (beg == end) {
        return end;
    }

    if constexpr (std::is_base_of_v<std::bidirectional_iterator_tag, ItCategory>) {
        return std::prev(end);
    } else {
        ItType prev_it = end;
        for (auto it = beg; it != end; ++it) {
            prev_it = it;
        }
        return prev_it;
    }
}

} // namespace estd
