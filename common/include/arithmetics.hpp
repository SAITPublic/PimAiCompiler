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

#include <limits>
#include <numeric>

#include "log.hpp"
#include "wrapper_types.hpp"

/// @brief returns result of integer division with rounding up: works as ceil( float(a) / b)
template <typename TInt1, typename TInt2>
ALWAYS_INLINE auto divUp(TInt1 numerator, TInt2 denominator) {
    static_assert(std::numeric_limits<std::common_type_t<TInt1, TInt2>>::is_integer, "Integer types are allowed only");
    constexpr bool is_signed = std::numeric_limits<TInt1>::is_signed || std::numeric_limits<TInt2>::is_signed;

    Log::COMMON::E_IF(denominator == 0) << "Zero division detected";
    Log::COMMON::E_IF(is_signed && (numerator == std::numeric_limits<TInt1>::min()) &&
                      (denominator == static_cast<TInt2>(-1)))
        << "Overflow detected";

    return (numerator + denominator - 1) / denominator;
}

/// @brief returns first argument aligned up (i.e. rounded up) by the second one
template <typename TInt1, typename TInt2>
inline auto alignUp(TInt1 value_to_be_aligned, TInt2 alignment_base) {
    static_assert(std::numeric_limits<std::common_type_t<TInt1, TInt2>>::is_integer, "Integer types are allowed only");
    return alignment_base * divUp(value_to_be_aligned, alignment_base);
}

/// @brief returns first argument aligned down (i.e. rounded down) by the second one
template <typename TInt1, typename TInt2>
inline auto alignDown(TInt1 value_to_be_aligned, TInt2 alignment_base) {
    static_assert(std::numeric_limits<std::common_type_t<TInt1, TInt2>>::is_integer, "Integer types are allowed only");
    Log::COMMON::E_IF(alignment_base == 0) << "Zero-base alignment detected";
    return value_to_be_aligned - (value_to_be_aligned % alignment_base);
}

/**
 * @brief Returns the sub-sequence last element coordinate provided the sub-secuence starts at given coordinate
 * @example
 *          [a][b][c][d]
 *          |<─-─-─-─->|─-─{ length = 4
 *          |┌-─-─-─-─-|─-─{ begin  = 3
 *          ||        ┌|─-─{ The value of interest is "coordinate of the last element"
 *          |v        v|
 * [0][1][2][3][4][5][6][7][8][9][...] coordinates (addresses/locations/indices)
 */
template <typename TInt1, typename TInt2>
inline auto getEndCoordinate(TInt1 begin, TInt2 length) {
    static_assert(std::numeric_limits<std::common_type_t<TInt1, TInt2>>::is_integer, "Integer types are allowed only");
    Log::COMMON::E_IF(length == 0) << "Invalid parameter was passed: length cannot be zero";
    return begin + length - 1;
}

/**
 * @brief Returns the length sub-sequence which begins and ends at given coordinates
 * @details This function is inverse version of getEndCoordinate
 */
template <typename TInt1, typename TInt2>
inline auto getLengthFromStartToEnd(TInt1 begin, TInt2 end) {
    static_assert(std::numeric_limits<std::common_type_t<TInt1, TInt2>>::is_integer, "Integer types are allowed only");
    Log::COMMON::E_IF(end < begin) << "Invalid parameters were passed: end shouldn't be less than begin";
    return end - begin + 1;
}

/// @brief Computes whether two closed intervals intersect
template <typename T>
bool doIntervalsIntersect(T l1, T r1, T l2, T r2) {
    return l1 <= r2 && l2 <= r1;
}

/// @brief returns size in bits given size in bytes
template <typename TInt>
inline uint64_t getSizeInBits(TInt size_in_bytes) {
    static_assert(std::numeric_limits<TInt>::is_integer, "Integer types are allowed only");
    // Use uint64_t to avoid overflows.
    return static_cast<uint64_t>(size_in_bytes) * static_cast<uint64_t>(CHAR_BIT);
}

/// @brief returns size in bytes given blob units number and blob unit size.
template <typename TInt>
inline uint64_t getSizeInBytes(TInt blob_units, nn_type::BlobUnitSize blob_unit_size) {
    static_assert(std::numeric_limits<TInt>::is_integer, "Integer types are allowed only");
    // Use uint64_t to avoid overflows.
    uint64_t size_in_bits = static_cast<uint64_t>(blob_units) * blob_unit_size.value();
    return divUp(size_in_bits, CHAR_BIT);
}

/// @brief returns result of unsigned subtraction while verifying that it does not wrap
template <typename TInt1, typename TInt2>
inline auto operator-(nn_type::SafeUInt<TInt1> lhs, nn_type::SafeUInt<TInt2> rhs) {
    Log::COMMON::E_IF(lhs.val < rhs.val) << "Unsigned wrap detected";
    return lhs.val - rhs.val;
}

/// @brief returns result of unsigned addition while verifying that it does not wrap
template <typename TInt1, typename TInt2>
inline auto operator+(nn_type::SafeUInt<TInt1> lhs, nn_type::SafeUInt<TInt2> rhs) {
    auto res = lhs.val + rhs.val;
    Log::COMMON::E_IF(res < lhs.val || res < rhs.val) << "Unsigned wrap detected";
    return res;
}

/// @brief returns size in bytes given size in bits
inline uint64_t getSizeInBytes(nn_type::BlobUnitSize size_in_bits) { return divUp(size_in_bits.value(), CHAR_BIT); }

/// @brief get absolute difference
template <typename T>
T absSub(T a, T b) {
    static_assert(std::numeric_limits<T>::is_integer, "Integer types are only allowed here");
    return a > b ? a - b : b - a;
}

template <typename TInt1, typename TInt2>
inline std::common_type_t<TInt1, TInt2> applyDilation(TInt1 length, TInt2 dilation) {
    static_assert(std::numeric_limits<std::common_type_t<TInt1, TInt2>>::is_integer, "Integer types are allowed only");
    return 1 + (length - 1) * dilation;
}
