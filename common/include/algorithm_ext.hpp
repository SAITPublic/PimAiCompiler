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

#include <algorithm>
#include <numeric>

namespace estd { // comes from 'extended std'

template <typename InputIteratorT, typename OutputIteratorT, typename TransformFunctionT, typename PredicateT>
auto transform_if(
    InputIteratorT first, InputIteratorT last, OutputIteratorT result, TransformFunctionT trans_func, PredicateT pred) {
    // TODO(r.rusyaev): implements compile-time check (similar to C++20 concepts)
    //                  to check that template parameters are valid
    for (; first != last; ++first) {
        if (pred(*first)) {
            *result = trans_func(*first);
            ++result;
        }
    }

    return result;
}

template <typename RangeT, typename OutputIteratorT, typename TransformFunctionT, typename PredicateT>
auto transform_if(RangeT&& range, OutputIteratorT result, TransformFunctionT trans_func, PredicateT pred) {
    return transform_if(std::begin(range), std::end(range), result, std::move(trans_func), std::move(pred));
}

template <typename RangeT, typename OutputIteratorT, typename TransformFunctionT>
auto transform(RangeT&& range, OutputIteratorT result, TransformFunctionT trans_func) {
    return std::transform(std::begin(range), std::end(range), result, trans_func);
}

template <typename RangeT, typename CompT>
void sort(RangeT&& range, CompT comp) {
    std::sort(std::begin(range), std::end(range), comp);
}

template <typename RangeT, typename CompT>
auto min_element(RangeT&& range, CompT comp) {
    return std::min_element(range.begin(), range.end(), comp);
}

template <typename RangeT, typename OutputIteratorT>
auto copy(RangeT&& range, OutputIteratorT result) {
    return std::copy(std::begin(range), std::end(range), result);
}

template <typename RangeT, typename OutputIteratorT, typename PredicateT>
auto copy_if(RangeT&& range, OutputIteratorT result, PredicateT pred) {
    return std::copy_if(std::begin(range), std::end(range), result, pred);
}

template <typename RangeT, typename UnaryFuncT>
auto for_each(RangeT&& range, UnaryFuncT func) {
    return std::for_each(std::begin(range), std::end(range), func);
}

template <typename RangeT, typename PredicateT>
bool any_of(RangeT&& range, PredicateT pred) {
    return std::any_of(std::begin(range), std::end(range), pred);
}

template <typename RangeT, typename PredicateT>
bool all_of(RangeT&& range, PredicateT pred) {
    return std::all_of(std::begin(range), std::end(range), pred);
}

template <typename RangeT, typename PredicateT>
bool none_of(RangeT&& range, PredicateT pred) {
    return std::none_of(std::begin(range), std::end(range), pred);
}

template <typename RangeT, typename T>
auto find(RangeT&& range, const T& key) {
    return std::find(std::begin(range), std::end(range), key);
}

template <typename RangeT, typename PredicateT>
auto find_if(RangeT&& range, PredicateT pred) {
    return std::find_if(std::begin(range), std::end(range), pred);
}

template <typename RangeT, typename PredicateT>
auto count_if(RangeT&& range, const PredicateT pred) {
    return std::count_if(std::begin(range), std::end(range), pred);
}

template <typename RangeT, typename PredicateT>
auto remove_if(RangeT&& range, const PredicateT pred) {
    return std::remove_if(std::begin(range), std::end(range), pred);
}

/// @brief checks if the vector is empty or contains only zeroes
template <typename T>
bool is_zero_container(T&& c) {
    return find_if(std::forward<T>(c), [](auto x) { return !!x; }) == c.end();
}

namespace detail {

template <typename ContainerT, typename T>
auto containsImpl(const ContainerT& cont, const T& val) -> decltype(cont.find(val), bool()) {
    return cont.find(val) != std::end(cont);
}

template <typename ContainerT, typename... T>
bool containsImpl(const ContainerT& cont, const T&... val) {
    static_assert(sizeof...(val) == 1,
                  "this overloading is needed only for SFINAE and must not have more than one parameter");
    return (find(cont, val), ...) != std::end(cont);
}

} // namespace detail

/// @brief returns true if `cont` contains `val`
template <typename ContainerT, typename T>
bool contains(const ContainerT& cont, const T& val) {
    return detail::containsImpl(cont, val);
}

template <typename ContainerT, typename InitT, typename OperationT>
InitT accumulate(const ContainerT& container, InitT init, OperationT operation) {
    return std::accumulate(container.begin(), container.end(), init, operation);
}

template <typename ContainerT, typename InitT>
InitT accumulate(const ContainerT& container, InitT init) {
    return std::accumulate(container.begin(), container.end(), init);
}

/// @brief a helper for estd::reverse
template <typename IterableT>
struct reversed_wrapper {
    IterableT& wrapped;
};

template <typename IterableT>
auto begin(reversed_wrapper<IterableT> wrapper) {
    return std::make_reverse_iterator(wrapper.wrapped.end());
}

template <typename IterableT>
auto end(reversed_wrapper<IterableT> wrapper) {
    return std::make_reverse_iterator(wrapper.wrapped.begin());
}

/// @brief enables efficient reversed-foreach loops
template <typename IterableT>
reversed_wrapper<IterableT> reverse(IterableT&& iterable) {
    return {iterable};
}

} // namespace estd
