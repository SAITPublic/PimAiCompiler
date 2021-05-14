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

#include "common/iterators.hpp"

namespace estd { // comes from extended std

/// @brief default traits for intrusive list. If you want to register
///        your own traits you have to define all static methods defined below
template <typename T>
struct IntrusiveListTraits {
    static void onAddNode(T& node) {}
    static void onDeleteNode(T& node) {}
    static void onMoveNodes(T& first, T& last) {}

    static std::string getPrintableNode(const T& node) { return {}; }

    constexpr static bool isListOwning() { return true; }
};

template <typename T>
struct NonOwningIntrusiveListTraits : IntrusiveListTraits<T> {
    constexpr static bool isListOwning() { return false; }
};

/// @brief This class represent intrusive list node. It provides the main API of intrusive list
///        node to get prev/next element. All users that want to use intrusive lists have to
///       inherit from this class via CRPT (it means if they are contained in intrusive list)
template <typename T>
class IntrusiveListNode {
 public:
    T* getPrev() { return static_cast<T*>(prev_); }
    T* getNext() { return static_cast<T*>(next_); }

    const T* getPrev() const { return static_cast<const T*>(prev_); }
    const T* getNext() const { return static_cast<const T*>(next_); }

    void swap(IntrusiveListNode& other) {
        prev_->next_       = &other;
        other.prev_->next_ = this;

        next_->prev_       = &other;
        other.next_->prev_ = this;

        std::swap(*this, other);
    }

    template <typename U, typename Traits>
    friend class intrusive_list;

 private:
    IntrusiveListNode<T>* prev_ = nullptr;
    IntrusiveListNode<T>* next_ = nullptr;
};

/// @brief intrusive list class contains elements (with or w/or ownership) and allows
///        users to register own hooks that will be called if new element
///        is added to list, removed or transferred from it
template <typename T, typename Traits = IntrusiveListTraits<T>>
class intrusive_list { // use snake_case to preserve naming convention with std containers
 private:
    using ListNode = IntrusiveListNode<T>;

 public:
    ///
    /** begin: constructors, destructor and special functions **/
    ///

    intrusive_list() : size_(0) {
        static_assert(std::is_base_of_v<ListNode, T>, "type of list must be derived from IntrusiveListNode");
        end_.prev_ = end_.next_ = &end_;
    }

    // prevent copy operations
    intrusive_list(const intrusive_list&) = delete;
    intrusive_list& operator=(const intrusive_list&) = delete;

    intrusive_list(intrusive_list&& other) { splice(end(), other, other.begin(), other.end()); }

    intrusive_list& operator=(intrusive_list&& other) {
        if (this != &other) {
            clear();
            splice(end(), other, other.begin(), other.end());
        }
        return *this;
    }

    ~intrusive_list() { clear(); }

    /** end: constructors, destructor and special functions **/

    ///
    /** begin: iterators **/
    ///

    template <bool is_const>
    class iterator_impl : public std::iterator<std::bidirectional_iterator_tag, T> {
     public:
        using reference = std::conditional_t<is_const, const T&, T&>;
        using pointer   = std::conditional_t<is_const, const T*, T*>;
        using NodeType  = std::conditional_t<is_const, const ListNode, ListNode>;

        iterator_impl() = default;
        explicit iterator_impl(NodeType* list_node) : node_(list_node) {}

        // allows implicit conversion from iterator to const_iterator
        template <bool is_other_const>
        iterator_impl(const iterator_impl<is_other_const>& const_iter,
                      std::enable_if_t<is_const || !is_other_const, void*> = nullptr)
            : node_(const_iter.node_) {}

        reference operator*() const { return *static_cast<pointer>(node_); }
        pointer   operator->() const { return &operator*(); }

        friend bool operator==(const iterator_impl& l, const iterator_impl& r) { return l.node_ == r.node_; }
        friend bool operator!=(const iterator_impl& l, const iterator_impl& r) { return !operator==(l, r); }

        iterator_impl& operator++() {
            node_ = node_->next_;
            return *this;
        }

        iterator_impl& operator--() {
            node_ = node_->prev_;
            return *this;
        }

        const iterator_impl operator++(int) {
            iterator_impl tmp = *this;
            ++*this;
            return tmp;
        }

        const iterator_impl operator--(int) {
            iterator_impl tmp = *this;
            --*this;
            return tmp;
        }

        NodeType* getListNode() const { return node_; }

     private:
        friend class iterator_impl<!is_const>;
        friend class intrusive_list;

        iterator_impl<false> getNonConstIterator() const { return iterator_impl<false>(const_cast<ListNode*>(node_)); }

        NodeType* node_ = nullptr;
    };

    using iterator               = iterator_impl<false>;
    using const_iterator         = iterator_impl<true>;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator       begin() { return iterator(end_.next_); }
    const_iterator begin() const { return const_iterator(end_.next_); }
    const_iterator cbegin() const { return begin(); }

    iterator       end() { return iterator(&end_); }
    const_iterator end() const { return const_iterator(&end_); }
    const_iterator cend() const { return end(); }

    reverse_iterator       rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const { return rbegin(); }

    reverse_iterator       rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const { return rend(); }

    /** end: iterators **/

    ///
    /** begin: accessors **/
    ///

    NO_DISCARD std::size_t size() const { return size_; }
    NO_DISCARD bool        empty() const { return size_ == 0; }

    T&       front() { return *begin(); }
    const T& front() const { return *begin(); }

    T&       back() { return *rbegin(); }
    const T& back() const { return *rbegin(); }

    /** end: accessors **/

    ///
    /** begin: add new elements **/
    ///

    iterator insert(const_iterator pos, const T& val) { return insertImpl(pos, val); }
    iterator insert(const_iterator pos, T&& val) { return insertImpl(pos, std::move(val)); }

    template <typename... ArgsT>
    iterator emplace(const_iterator pos, ArgsT&&... args) {
        return insertImpl(pos, std::forward<ArgsT>(args)...);
    }

    void push_front(const T& val) { insertImpl(begin(), val); }
    void push_front(T&& val) { insertImpl(begin(), std::move(val)); }

    void push_back(const T& val) { insertImpl(end(), val); }
    void push_back(T&& val) { insertImpl(end(), std::move(val)); }

    template <typename... ArgsT>
    void emplace_front(ArgsT&&... args) {
        insertImpl(begin(), std::forward<ArgsT>(args)...);
    }

    template <typename... ArgsT>
    void emplace_back(ArgsT&&... args) {
        insertImpl(end(), std::forward<ArgsT>(args)...);
    }

    iterator insert(const_iterator pos, std::unique_ptr<T> val) {
        auto* val_ptr = val.release();

        Traits::onAddNode(*val_ptr);
        linkNodes(pos.getNonConstIterator().getListNode(), val_ptr);
        ++size_;

        return iterator(val_ptr);
    }

    void push_back(std::unique_ptr<T> val) { insert(end(), std::move(val)); }
    void push_front(std::unique_ptr<T> val) { insert(begin(), std::move(val)); }

    /** end: add new elements **/

    ///
    /** begin: delete elements **/
    ///

    iterator erase(const_iterator pos) {
        auto next_pos = std::next(pos.getNonConstIterator());

        auto* rem_node = pos.getNonConstIterator().getListNode();

        auto* prev = rem_node->prev_;
        auto* next = rem_node->next_;

        next->prev_ = prev;
        prev->next_ = next;

        --size_;

        // down cast is necessary because destructor of ListNode is not virtual
        auto* val = static_cast<T*>(rem_node);

        Traits::onDeleteNode(*val);

        if (Traits::isListOwning()) {
            delete val;
        }

        return next_pos;
    }

    void erase(const_iterator first, const_iterator last) {
        for (; first != last; first = erase(first)) {
        }
    }

    void clear() { erase(begin(), end()); }

    /// @brief remove all elements from the list if they satisfy predicate
    template <typename PredT>
    void remove_if(PredT pred) {
        for (auto it = begin(), end = this->end(); it != end;) {
            if (pred(*it)) {
                it = erase(it);
            } else {
                ++it;
            }
        }
    }

    /** end: delete elements **/

    ///
    /** begin: transfer/permute elements **/
    ///

    void swap(intrusive_list& other) {
        end_.swap(other.end_);
        std::swap(size_, other.size_);
    }

    /// @brief transfers elements from 'other' list to this one
    /// @param next_pos -- insert before this element
    /// @param first -- start transferring from this element
    /// @param after_last -- stop transferring before this element
    void splice(const_iterator next_pos, intrusive_list& other, const_iterator first, const_iterator after_last) {
        if (first == after_last) {
            return;
        }

        auto last = --after_last;

        Log::COMMON::E_IF(this == &other) << "invalid splice operation";

        auto moved_nodes_num = std::distance(first, last) + 1;

        auto first_node    = first.getNonConstIterator().getListNode();
        auto last_node     = last.getNonConstIterator().getListNode();
        auto next_pos_node = next_pos.getNonConstIterator().getListNode();

        // call registered hook
        Traits::onMoveNodes(*static_cast<T*>(first_node), *static_cast<T*>(last_node));

        // unlink nodes from other list and correct size
        first_node->prev_->next_ = last_node->next_;
        last_node->next_->prev_  = first_node->prev_;
        other.size_ -= moved_nodes_num;

        // link moved nodes to this list and increase list size
        next_pos_node->prev_->next_ = first_node;
        first_node->prev_           = next_pos_node->prev_;
        next_pos_node->prev_        = last_node;
        last_node->next_            = next_pos_node;
        size_ += moved_nodes_num;
    }

    /// @brief merge sorted 'other' list into 'this' sorted list
    template <typename CompT>
    void merge(intrusive_list& other, CompT comp) {
        if (this == &other || other.empty()) {
            return;
        }

        auto other_it  = other.begin();
        auto this_end  = end();
        auto other_end = other.end();

        for (auto it = begin(); it != this_end; ++it) {
            if (!comp(*other_it, *it)) {
                // element from other list is not less than element from this list
                continue;
            }

            auto old_other_it = other_it;

            // find the first element in other list that is not less than cur element in this list
            other_it = std::find_if(std::next(other_it), other.end(), [&it, &comp](auto& v) { return !comp(v, *it); });

            splice(it, other, old_other_it, other_it);

            if (other_it == other_end) {
                return;
            }
        }

        splice(this_end, other, other_it, other_end);
    }

    /// @brief sort list elements using custom comparator
    template <typename CompT>
    void sort(CompT comp) {
        if (size_ == 0 || size_ == 1) {
            // nothing to sort
            return;
        }

        // find the middle of the list
        auto mid = begin();
        std::advance(mid, size_ / 2);

        intrusive_list<T, NonOwningIntrusiveListTraits<T>> list;
        auto&                                              right_half = static_cast<intrusive_list&>(list);

        right_half.splice(right_half.end(), *this, mid, end()); // obtain right half of the list

        sort(comp);                     // sort left half
        right_half.template sort(comp); // sort right half
        merge(right_half, comp);        // merge 'this' (left half) with right half
    }

    /** end: transfer/permute elements **/

    ///
    /** begin: intrusive list specific **/
    ///

    iterator getNodeIterator(T& item) {
        auto it = iterator(&item);
        checkIteratorCorrectness(it);
        return it;
    }

    const_iterator getNodeIterator(const T& item) const {
        auto it = const_iterator(&item);
        checkIteratorCorrectness(it);
        return it;
    }

    /** end: intrusive list specific **/

 private:
    template <typename... ArgsT>
    iterator insertImpl(const_iterator pos, ArgsT&&... args) {
        T* new_node_p = newNode(std::forward<ArgsT>(args)...);
        Traits::onAddNode(*new_node_p);

        linkNodes(pos.getNonConstIterator().getListNode(), new_node_p);
        ++size_;

        return iterator(new_node_p);
    }

    template <typename... ArgsT>
    T* newNode(ArgsT&&... args) {
        static_assert(
            Traits::isListOwning(),
            "It has to be used `insert` method that takes unique_ptr to insert new element in non-owning list");

        return new T(std::forward<ArgsT>(args)...);
    }

    void linkNodes(ListNode* next_node, ListNode* new_node) {
        auto* prev_node = next_node->prev_;

        //      |---> |----------| <---|
        //      | |---| new_node |---| |
        //      | |   |----------|   | |
        //      | V                  V |
        // |-----------|  ----> |-----------| ---->
        // | prev_node |        | next_node |      ...
        // |-----------|  <---- |-----------| <----

        prev_node->next_ = new_node;
        new_node->prev_  = prev_node;

        next_node->prev_ = new_node;
        new_node->next_  = next_node;
    }

    void checkIteratorCorrectness(const_iterator it_for_check) const {
#ifndef NDEBUG
        for (auto it = begin(); it != end(); ++it) {
            if (it == it_for_check) {
                return;
            }
        }
        Log::COMMON::E() << "Invalid iterator for list node: " << Traits::getPrintableNode(*it_for_check);
#endif // NDEBUG
    }

    template <typename U, typename UT>
    friend class intrusive_list;

    // necessary in sort algorithm to cast list with NonOwningTraits to list with owning traits
    // it can be useful when we need non-owing list for some internal stuff (to not destroy list objects),
    // Note that this conversion is unsafe in common case. For example if this cast happens in context
    // of owning operations (i.e erase) we can get issues like memory leak. For this reason that
    // conversion operator has to stay always private and must be used explicitly
    template <typename OtherTraits>
    explicit operator intrusive_list<T, OtherTraits>&() {
        return reinterpret_cast<intrusive_list<T, OtherTraits>&>(*this);
    }

    // end_ itself marks the end of the list
    //   end_.next_ -- points to the first element of the list
    //   end_.prev_ -- points to the last element of the list
    ListNode end_;
    // contains current number of elements in list
    std::size_t size_;
};

} // namespace estd

namespace std {

template <typename T>
void swap(estd::intrusive_list<T>& l, estd::intrusive_list<T>& r) {
    l.swap(r);
}

template <typename T>
void swap(estd::IntrusiveListNode<T>& l, estd::IntrusiveListNode<T>& r) {
    l.swap(r);
}

} // namespace std
