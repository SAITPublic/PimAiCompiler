/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    iterators.hpp
 * @brief.   Various classes to simplify iteration.
 */

#pragma once

#include <type_traits>

template <typename It>
class iterator_range {
    It begin_, end_;

 public:
    using iterator = It;

    iterator_range(It begin, It end) : begin_(begin), end_(end) {}

    It begin() const { return begin_; }

    It end() const { return end_; }

    bool empty() const { return begin_ == end_; }
};

/**
 * @brief: A very simple implementation of 'Iterator Facade' idiom from boost, needed to simplify custom
 *         iterators creation. See https://www.boost.org/doc/libs/1_65_0/libs/iterator/doc/iterator_facade.html
 *         for the detailed discussion about the idiom.
 *         The user (Derived class) must implement the following operations:
 *           ValueT& dereference() const;       -- dereference iterator to get an underlying value
 *           bool equal(const Derived&) const;  -- return true iff two iterators refer to the same position
 *           void increment();                  -- increment iterator
 *           void decrement();                  -- decrement iterator (only for bidirectional or random
 *                                                 access iterators)
 *           Derived& advance(DiffT n);         -- advance iterator by 'n' positions ('n' can be a negative number;
 *                                                 only for random access iterators)
 *           DiffT distance_to(const Derived&); -- get distance between two iterators (only for random access iterators)
 */
template <typename Derived,
          typename ValueT,
          typename ItCategoryT,
          typename DiffT = ptrdiff_t,
          typename PtrT  = ValueT*,
          typename RefT  = ValueT&>
class IteratorFacade : public std::iterator<ItCategoryT, ValueT, DiffT, PtrT, RefT> {
 private:
    static constexpr bool is_random   = std::is_same_v<std::random_access_iterator_tag, ItCategoryT>;
    static constexpr bool is_bidirect = std::is_base_of_v<std::bidirectional_iterator_tag, ItCategoryT>;
    static constexpr bool is_forward  = std::is_base_of_v<std::forward_iterator_tag, ItCategoryT>;

 public:
    Derived operator+(DiffT n) {
        static_assert(std::is_base_of_v<IteratorFacade, Derived>, "must be used via CRTP");
        static_assert(is_random, "'+' is allowed only for random access iterator");

        return static_cast<Derived*>(this)->advance(n);
    }

    Derived operator-(DiffT n) {
        static_assert(std::is_base_of_v<IteratorFacade, Derived>, "must be used via CRTP");
        static_assert(is_random, "'-' is allowed only for random access iterator");

        return static_cast<Derived*>(this)->advance(-n);
    }

    DiffT operator-(const Derived& rhs) const {
        static_assert(std::is_base_of_v<IteratorFacade, Derived>, "must be used via CRTP");
        static_assert(is_random, "'-' is allowed only for random access iterator");

        return static_cast<const Derived*>(this)->distance_to(rhs);
    }

    Derived& operator+=(DiffT n) {
        static_assert(std::is_base_of_v<IteratorFacade, Derived>, "must be used via CRTP");
        static_assert(is_random, "'+=' is allowed only for random access iterator");

        return static_cast<Derived*>(this)->advance(n);
    }

    Derived& operator-=(DiffT n) {
        static_assert(std::is_base_of_v<IteratorFacade, Derived>, "must be used via CRTP");
        static_assert(is_random, "'-=' is allowed only for random access iterator");

        return static_cast<Derived*>(this)->advance(-n);
    }

    Derived& operator++() {
        static_assert(std::is_base_of_v<IteratorFacade, Derived>, "must be used via CRTP");
        static_assert(is_forward, "'++' is allowed only for random access, bidirectional or forward iterators");

        static_cast<Derived*>(this)->increment();
        return *static_cast<Derived*>(this);
    }

    Derived operator++(int) {
        auto tmp(static_cast<Derived* const>(this));
        ++*this;
        return *tmp;
    }

    Derived& operator--() {
        static_assert(std::is_base_of_v<IteratorFacade, Derived>, "must be used via CRTP");
        static_assert(is_bidirect, "'--' is allowed only for random access or bidirectional iterators");

        static_cast<Derived*>(this)->decrement();
        return *static_cast<Derived*>(this);
    }

    Derived operator--(int) {
        auto tmp(static_cast<Derived* const>(this));
        --*this;
        return *tmp;
    }

    ValueT& operator*() const { return static_cast<Derived const*>(this)->dereference(); }

    ValueT* operator->() const { return &(static_cast<Derived const*>(this)->dereference()); }

    friend bool operator==(const IteratorFacade& lhs, const IteratorFacade& rhs) {
        return static_cast<const Derived&>(lhs).equal(static_cast<const Derived&>(rhs));
    }

    friend bool operator!=(const IteratorFacade& lhs, const IteratorFacade& rhs) {
        return !static_cast<const Derived&>(lhs).equal(static_cast<const Derived&>(rhs));
    }
};

/**
 * @brief: A very simple implementation of 'Iterator Adaptor' idiom from boost, needed to simplify custom
 *         iterators creation and reducing code boilerplate on common operations (e.g. increment, equal etc).
 *         See https://www.boost.org/doc/libs/1_65_0/libs/iterator/doc/iterator_adaptor.html
 *         for the detailed discussion about the idiom.
 */
template <typename Derived,
          typename BaseT,
          typename ValueT      = typename std::iterator_traits<BaseT>::value_type,
          typename ItCategoryT = typename std::iterator_traits<BaseT>::iterator_category,
          typename DiffT       = typename std::iterator_traits<BaseT>::difference_type,
          typename PtrT = std::conditional_t<std::is_same_v<ValueT, typename std::iterator_traits<BaseT>::value_type>,
                                             typename std::iterator_traits<BaseT>::pointer,
                                             ValueT*>,
          typename RefT = std::conditional_t<std::is_same_v<ValueT, typename std::iterator_traits<BaseT>::value_type>,
                                             typename std::iterator_traits<BaseT>::reference,
                                             ValueT&>>
class IteratorAdaptor : public IteratorFacade<Derived, ValueT, ItCategoryT, DiffT, PtrT, RefT> {
 public:
    IteratorAdaptor() = default; // for STL algorithms
    explicit IteratorAdaptor(const BaseT& iter) : m_iterator_(iter) {}

    void increment() { ++m_iterator_; }

    void decrement() { --m_iterator_; }

    ValueT& dereference() const { return *m_iterator_; }

    Derived& advance(DiffT n) {
        m_iterator_ += n;
        return *static_cast<Derived*>(this);
    }

    DiffT distance_to(const Derived& rhs) const { return m_iterator_ - rhs.m_iterator_; }

    bool equal(const Derived& rhs) const { return m_iterator_ == rhs.m_iterator_; }

    BaseT& base() { return m_iterator_; }

    const BaseT& base() const { return m_iterator_; }

 private:
    BaseT m_iterator_;
};

/*
 * @brief: Iterator for filtering values acquired from iterator of base type
 */
template <typename ValueT, typename BaseT, typename FilterTraits>
class FilterIterator : public IteratorAdaptor<FilterIterator<ValueT, BaseT, FilterTraits>, BaseT, ValueT> {
    using Super = IteratorAdaptor<FilterIterator<ValueT, BaseT, FilterTraits>, BaseT, ValueT>;
    BaseT end_guard_;

 public:
    FilterIterator(const BaseT& base, const BaseT& end_guard) : Super(base), end_guard_(end_guard) {}

    void increment() {
        auto& base_iterator = this->base();
        do {
            ++base_iterator;
        } while (base_iterator != end_guard_ && !FilterTraits::isKindOf(*base_iterator));
    }

    ValueT& dereference() const { return FilterTraits::cast(*this->base()); }
}; // class FilterIterator

/*
 * @brief: Iterator for filtering values acquired from iterator of base type
 */
template <typename ValueT, typename BaseT, typename PredT, typename FilterTraits>
class UnaryPredFilterIterator
    : public IteratorAdaptor<UnaryPredFilterIterator<ValueT, BaseT, PredT, FilterTraits>, BaseT, ValueT> {

    using Super = IteratorAdaptor<UnaryPredFilterIterator<ValueT, BaseT, PredT, FilterTraits>, BaseT, ValueT>;
    BaseT                                                                                end_guard_;
    std::conditional_t<std::is_function<PredT>::value, std::add_pointer_t<PredT>, PredT> pred_;

 public:
    UnaryPredFilterIterator(const BaseT& base, const BaseT& end_guard, const PredT& pred)
        : Super(base), end_guard_(end_guard), pred_(pred) {}

    void increment() {
        auto& base_iterator = this->base();
        do {
            ++base_iterator;
        } while (base_iterator != this->end_guard_ && !FilterTraits::isKindOf(*base_iterator, pred_));
    }

    ValueT& dereference() const {
        using PureValueT = typename std::remove_cv<typename std::remove_reference<ValueT>::type>::type;
        using PureBaseT =
            typename std::remove_cv<typename std::remove_reference<typename BaseT::value_type>::type>::type;
        if constexpr (std::is_same_v<PureBaseT, PureValueT>) {
            return *this->base();
        } else {
            return FilterTraits::cast(*this->base());
        }
    }
}; // class UnaryPredFilterIterator

/*
 * @brief: Abstraction for default filter iterator traits
 */
template <typename ValueT>
struct DefaultFilterIteratorTraits;

/*
 * @brief: Obtains iterator range of filtered values
 */
template <typename ValueT, typename IterableT, typename FilterTraits = DefaultFilterIteratorTraits<ValueT>>
auto makeFilteredRange(const IterableT& range)
    -> iterator_range<FilterIterator<ValueT, decltype(std::begin(range)), FilterTraits>> {
    auto begin = std::begin(range);
    auto end   = std::end(range);
    while (begin != end && !FilterTraits::isKindOf(*begin)) {
        ++begin;
    }
    return {{begin, end}, {end, end}};
}

/*
 * @brief: Obtains iterator range of filtered values basing on value type and provided unary predicate.
 * @param[in]: range A source range to generate desired range of typed, predicated values.
 * @param[in]: pred An unary predicate that decides whether current value is suitable for resulting range
 *             or not.
 */
template <typename ValueT,
          typename IterableT,
          typename PredT,
          typename FilterTraits = DefaultFilterIteratorTraits<ValueT>>
auto makeUnaryPredFilteredRange(const IterableT& range, const PredT& pred)
    -> iterator_range<UnaryPredFilterIterator<ValueT, decltype(std::begin(range)), PredT, FilterTraits>> {
    auto begin = std::begin(range);
    auto end   = std::end(range);
    while (begin != end && !FilterTraits::isKindOf(*begin, pred)) {
        ++begin;
    }
    return {{begin, end, pred}, {end, end, pred}};
}
