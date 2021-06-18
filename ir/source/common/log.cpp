/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    log.hpp
 * @brief.
 * @details. This defines Logger class methods.
 * @version. 0.1.
 */

#include "common/common.hpp"
#include "common/types.hpp"

#include "common/log.hpp"

#ifdef __linux__
#include <unistd.h>
#endif

namespace Log {

DebugLevelType debug_level_ = DebugLevelType::FULL;

}

void Logger::error(const char* message, ...) {
    va_list args;
    __tag(ANSI_COLOR_RED, LOGE_TAG);
    if (tag_)
        __tag(ANSI_COLOR_RED, tag_);
    va_start(args, message);
    __log(message, args);
    va_end(args);
}
void Logger::info(const char* message, ...) {
    va_list args;
    if (Log::debug_level_ != DebugLevelType::NONE) {
        __tag(ANSI_COLOR_GREEN, LOGI_TAG);
        if (tag_)
            __tag(ANSI_COLOR_GREEN, tag_);
        va_start(args, message);
        __log(message, args);
        va_end(args);
    }
}
void Logger::debug(const char* message, ...) {
    va_list args;
    if (Log::debug_level_ == DebugLevelType::FULL) {
        __tag(ANSI_COLOR_BLUE, LOGD_TAG);
        if (tag_)
            __tag(ANSI_COLOR_BLUE, tag_);
        va_start(args, message);
        __log(message, args);
        va_end(args);
    }
}
void Logger::temp(const char* message, ...) {
    va_list args;
    {
        __tag(ANSI_COLOR_CYAN, LOGT_TAG);
        if (tag_)
            __tag(ANSI_COLOR_CYAN, tag_);
        va_start(args, message);
        __log(message, args);
        va_end(args);
    }
}

void Logger::__log(const char* message, va_list& args) { vfprintf(stdout, message, args); }
void Logger::__tag(const char* color, const char* message) {
#ifdef __linux__
    static bool colorize = isatty(STDOUT_FILENO);
    if (colorize)
        fputs(color, stdout);
    fputs(message, stdout);
    if (colorize)
        fputs(ANSI_COLOR_RESET, stdout);
#else  // __linux__
    fputs(message, stdout);
#endif // __linux__
}
