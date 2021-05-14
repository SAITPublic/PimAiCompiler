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

#if __cplusplus >= 201703L // >= C++17
// [[nodiscard]] attribute: https://en.cppreference.com/w/cpp/language/attributes/nodiscard
#define NO_DISCARD [[nodiscard]] // NOLINT
// [[fallthrough]] attribute: https://en.cppreference.com/w/cpp/language/attributes/fallthrough
#define FALLTHROUGH [[fallthrough]] // NOLINT
// [[noreturn]] attribute: https://en.cppreference.com/w/cpp/language/attributes/noreturn
#define NO_RETURN [[noreturn]] // NOLINT
#else
#define NO_DISCARD
#define FALLTHROUGH
#define NO_RETURN
#endif

#ifndef __has_attribute
#define __has_attribute(attr) false
#endif

#ifndef __has_builtin
#define __has_builtin(built) false
#endif

#ifndef __has_feature
#define __has_feature(f) false
#endif

#if __has_builtin(__builtin_expect)
#define LIKELY(COND)   __builtin_expect(!!(COND), 1)
#define UNLIKELY(COND) __builtin_expect((COND), 0)
#else
#define LIKELY(COND)   (COND)
#define UNLIKELY(COND) (COND)
#endif

#if __has_attribute(always_inline)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif
