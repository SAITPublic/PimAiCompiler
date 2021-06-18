/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 u transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "log.hpp"
#include <climits>

namespace nn_type {

/** @brief A simple wrapper class around uint32_t to carry blobs unit size in bits.
 *  @details The main motivation for this is to prohibit all error prone blob->getUnitSize()
 *           comparisons/manipulations and use explicit getSizeInBytes on unit size.
 *           Since the class is a POD with single numeric value, instances of this class
 *           should be passed by value in function parameters.
 */
class BlobUnitSize {
 public:
    explicit BlobUnitSize(uint32_t size_in_bits) : size_in_bits_(size_in_bits) {}
    explicit BlobUnitSize(uint64_t size_in_bits) : size_in_bits_(size_in_bits) {}
    explicit BlobUnitSize(int32_t size_in_bits) : size_in_bits_(size_in_bits) {
        Log::COMMON::E_IF(size_in_bits < 0) << "Size in bits should be positive!";
    }
    explicit BlobUnitSize(int64_t size_in_bits) : size_in_bits_(size_in_bits) {
        Log::COMMON::E_IF(size_in_bits < 0) << "Size in bits should be positive!";
    }
    BlobUnitSize() : size_in_bits_(CHAR_BIT) {}
    uint64_t value() const { return size_in_bits_; }

 private:
    uint64_t size_in_bits_;
};

inline std::ostream& operator<<(std::ostream& s, BlobUnitSize size) {
    s << size.value();
    return s;
}

/// @brief simple wrapper around integral value that is used in safe arithmetic operations like +, - etc
template <typename T>
struct SafeInt {
    explicit SafeInt(T val) : val(val) {
        static_assert(std::is_integral_v<T>, "parameter type of SafeInt must be integral");
    }

    T val;
};

/// @brief simple wrapper around signed integer that is used in safe arithmetic operations like +, - etc
template <typename T>
struct SafeSInt {
    explicit SafeSInt(T val) : val(val) {
        static_assert(std::is_signed_v<T>, "parameter type of SafeSInt must be signed integer");
    }

    T val;
};

/// @brief simple wrapper around unsigned integer that is used in safe arithmetic operations like +, - etc
template <typename T>
struct SafeUInt {
    explicit SafeUInt(T val) : val(val) {
        static_assert(std::is_unsigned_v<T>, "parameter type of SafeUInt must be unsigned integer");
    }

    T val;
};

} // end of namespace nn_type
