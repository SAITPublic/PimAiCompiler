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

#include <type_traits>

#include "common/attributes.h"
#include "common/log.hpp"

/// these function implement type casting support for IR classes.
/// The contract for classes that will be used in down-casting has to be the following:
///  * class has to implement `static bool classof(Base*)` function
///
/// Examples of usage:
/// if (isa<NNode>(node) {
///     // actions
/// }
/// auto conv_node = cast<ConvolutionNode>(node);
/// if (conv_node)
///    ...
/// else
///   ...
///

template <typename To, typename From>
inline bool isa(const From& val) {
    return To::template classof<To>(&val);
}

template <typename To, typename From>
inline bool isa(const From* val) {
    return To::template classof<To>(val);
}

template <typename To, typename From>
inline bool isa(From* val) {
    return To::template classof<To>(val);
}

template <typename To1, typename To2, typename... ToN, typename From>
inline bool is_any_of(const From& val) {
    return ((isa<To1>(val) || isa<To2>(val)) || ... || isa<ToN>(val));
}

template <typename To, typename From>
NO_DISCARD typename std::add_const<To>::type* cast_if(const From& val) {
    using constTo = typename std::add_const<To>::type;
    return isa<To>(val) ? static_cast<constTo*>(&val) : nullptr;
}

template <typename To, typename From>
NO_DISCARD To* cast_if(From& val) {
    return isa<To>(val) ? static_cast<To*>(&val) : nullptr;
}

template <typename To, typename From>
NO_DISCARD typename std::add_const<To>::type* cast_if(const From* val) {
    using constTo = typename std::add_const<To>::type;
    return val && isa<To>(val) ? static_cast<constTo*>(val) : nullptr;
}

template <typename To, typename From>
NO_DISCARD To* cast_if(From* val) {
    return val && isa<To>(val) ? static_cast<To*>(val) : nullptr;
}

template <typename To, typename From>
typename std::add_const<To>::type& cast(const From& val) {
    auto res = cast_if<To>(val);
    Log::COMMON::E_IF(!res) << "Can't cast";
    return *res;
}

template <typename To, typename From>
To& cast(From& val) {
    auto res = cast_if<To>(val);
    Log::COMMON::E_IF(!res) << "Can't cast";
    return *res;
}

template <typename To, typename From>
typename std::add_const<To>::type& cast(const From* val) {
    auto res = cast_if<To>(val);
    Log::COMMON::E_IF(!res) << "Can't cast";
    return *res;
}

template <typename To, typename From>
To& cast(From* val) {
    auto res = cast_if<To>(val);
    Log::COMMON::E_IF(!res) << "Can't cast";
    return *res;
}
