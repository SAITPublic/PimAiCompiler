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

#include <functional>
#include <type_traits>

namespace ctti {

class TypeIndex {
 public:
    using IdType = char;

    explicit TypeIndex(IdType* id) : id(id) {}

    // define <, >, = etc operators to allow containing this class in associated containers
    bool operator==(const TypeIndex& ti) const { return id == ti.id; }
    bool operator!=(const TypeIndex& ti) const { return id != ti.id; }
    bool operator<(const TypeIndex& ti) const { return id < ti.id; }
    bool operator<=(const TypeIndex& ti) const { return id <= ti.id; }
    bool operator>(const TypeIndex& ti) const { return id > ti.id; }
    bool operator>=(const TypeIndex& ti) const { return id >= ti.id; }

    std::size_t getHash() const { return std::hash<IdType*>()(id); }

 private:
    IdType* id;
};

template <typename T>
TypeIndex typeIdWithCVR() {
    static TypeIndex::IdType id;
    return TypeIndex(&id);
}

template <typename T>
TypeIndex typeId() {
    using PureT = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    return typeIdWithCVR<PureT>();
}

} // namespace ctti

namespace std {

template <>
struct hash<ctti::TypeIndex> {
    std::size_t operator()(const ctti::TypeIndex& ti) const { return ti.getHash(); }
};

} // namespace std
