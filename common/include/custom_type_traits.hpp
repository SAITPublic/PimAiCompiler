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

#include <type_traits>
#include <utility>

namespace nn_compiler {
template <bool... Bs>
using bool_sequence = std::integer_sequence<bool, Bs...>;

template <bool... Bs>
using bool_and = std::is_same<bool_sequence<Bs...>, bool_sequence<(Bs || true)...>>;

template <bool... Bs>
using bool_or = std::integral_constant<bool, !bool_and<!Bs...>::value>;

template <typename T, typename... Ts>
using any_of = bool_or<std::is_same<T, Ts>::value...>;
} // namespace nn_compiler
