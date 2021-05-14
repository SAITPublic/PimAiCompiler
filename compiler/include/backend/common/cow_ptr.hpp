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

#include <memory>

namespace estd {

/// @brief this class provides lazy coping of object (copy-on-write)
template <typename ObjT>
class cow_ptr {
 public:
    explicit cow_ptr(std::shared_ptr<ObjT> obj_ptr) : ptr_(std::move(obj_ptr)) {}
    cow_ptr() { ptr_ = std::make_shared<ObjT>(); }

    const ObjT& operator*() const { return *ptr_; }

    ObjT& operator*() {
        detach();
        return *ptr_;
    }

    const ObjT* operator->() const { return ptr_.operator->(); }

    ObjT* operator->() {
        detach();
        return ptr_.operator->();
    }

 private:
    void detach() {
        if (ptr_.use_count() > 1) {
            // make copy of shared object (COW)
            ptr_ = std::make_shared<ObjT>(*ptr_);
        }
    }

    std::shared_ptr<ObjT> ptr_;
};

template <typename TObj, typename... ArgsT>
cow_ptr<TObj> make_cow(ArgsT&&... args) {
    return cow_ptr(std::make_shared<TObj>(args...));
}

} // namespace estd
