#pragma once

#include "common/include/common.hpp"
#include "common/include/log.hpp"

DEFINE_LOGGER(NC, "[NNCompiler]")
DEFINE_LOGGER(FE, "[FrontEnd]")
DEFINE_LOGGER(ME, "[MiddleEnd]")
DEFINE_LOGGER(BE, "[BackEnd]")

/*
 * This definition is legacy and obsolete. Please do not use it
 * and aid migration to the new C++ API; use Log::TAG::E_IF() instead.
 */
#define LOGE_IF(TAG, expr, fmt, ...)                                                \
    do {                                                                            \
        if (expr) {                                                                 \
            if constexpr (std::size(fmt) == 1 || fmt[std::size(fmt) - 2] != '\n') { \
                Log::TAG::getLogger().error(LOG_PREFIX() fmt "\n", ##__VA_ARGS__);  \
            } else {                                                                \
                Log::TAG::getLogger().error(LOG_PREFIX() fmt, ##__VA_ARGS__);       \
            }                                                                       \
            exit(1);                                                                \
        }                                                                           \
    } while (0)
