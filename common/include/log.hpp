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
 * @details. This header defines interfaces for log
 * @version. 0.1.
 */

#pragma once

#include "common/include/attributes.h"
#include "common/include/common.hpp"
#include "common/include/types.hpp"
#include <cstdarg>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define LOG_STRINGIZE2(X) #X
#define LOG_STRINGIZE(X)  LOG_STRINGIZE2(X)

#ifndef NDEBUG
#define LOG_PREFIX(X) X __FILE__ ":" LOG_STRINGIZE(__LINE__) ": "
#else
#define LOG_PREFIX(X) X
#endif

#define NN_DEBUG_IF(COND, X)                                       \
    do {                                                           \
        if (Log::debug_level_ == DebugLevelType::FULL && (COND)) { \
            X;                                                     \
        }                                                          \
    } while (0)
#define NN_INFO_IF(COND, X)                                        \
    do {                                                           \
        if (Log::debug_level_ != DebugLevelType::NONE && (COND)) { \
            X;                                                     \
        }                                                          \
    } while (0)
#define NN_ERROR_IF(COND, X) \
    do {                     \
        if (COND) {          \
            X;               \
        }                    \
    } while (0)

#define NN_DEBUG(X) NN_DEBUG_IF(true, X)
#define NN_INFO(X)  NN_INFO_IF(true, X)

#define LOGE_TAG "[LOGE] "
#define LOGI_TAG "[LOGI] "
#define LOGD_TAG "[LOGD] "
#define LOGT_TAG "[LOGT] "

namespace Log {
extern DebugLevelType debug_level_;
}

class Logger {
 public:
    explicit Logger(const char* tag = nullptr) : tag_(tag) {}

    void error(const char* message, ...);
    void info(const char* message, ...);
    void debug(const char* message, ...);
    void temp(const char* message, ...);

 protected:
    static void __log(const char* message, va_list& args);
    static void __tag(const char* color, const char* message);

 private:
    const char* tag_;
};

// internal stuff for LogStream. It allows to use custom overloaded operator<< in scoped namespaces
namespace Log {

template <typename T>
auto printImpl(std::ostringstream& os, const T& val, long) -> decltype(os.operator<<(val), void()) {
    os << val;
}

template <typename T>
auto printImpl(std::ostringstream& os, const T& val, int) -> decltype(operator<<(os, val), void()) {
    os << val;
}

template <typename T>
void printImpl(std::ostringstream& os, const T& val, ...) {
    std::ostream& operator<<(std::ostream& s, const T&);

    operator<<(os, val);
}

} // namespace Log

/// @brief Logger class that allows to print debug and error messages
/// @note If you have problems with ADL in case of custom 'operator<<' you need to define your
///       own 'operator<<' in Log namespace to prevent ADL in this case
class LogStream {
 public:
    explicit LogStream(Logger& logger, bool enabled) : logger_(logger), enabled_(enabled) {}
    LogStream(LogStream&& orig) = default;

    template <typename T>
    LogStream& operator<<(const T& val) {
        if (enabled_) {
            Log::printImpl(os_, val, 1L);
        }
        return *this;
    }

    Logger& getLogger() { return logger_; }

    bool isEnabled() const { return enabled_; }

 protected:
    Logger&            logger_;
    const bool         enabled_;
    std::ostringstream os_;
};

class ErrStream : public LogStream {
 public:
    explicit ErrStream(Logger& logger) : LogStream(logger, true) {}
    ErrStream(ErrStream&& orig) = default;

    NO_RETURN ~ErrStream() {
        logger_.error("%s\n", os_.str().c_str());
        exit(1);
    }
};

class ConditionalErrStream : public LogStream {
 public:
    explicit ConditionalErrStream(Logger& logger, bool exit_condition)
        : LogStream(logger, exit_condition), exit_condition_(exit_condition) {}
    ConditionalErrStream(ConditionalErrStream&& orig) = default;

    ALWAYS_INLINE ~ConditionalErrStream() {
        if (UNLIKELY(exit_condition_)) {
            logger_.error("%s\n", os_.str().c_str());
            exit(1);
        }
    }

 private:
    const bool exit_condition_;
};

class InfoStream : public LogStream {
 public:
    explicit InfoStream(Logger& logger) : LogStream(logger, Log::debug_level_ >= DebugLevelType::PARTIAL) {}
    InfoStream(InfoStream&& orig) = default;
    ~InfoStream() { logger_.info("%s\n", os_.str().c_str()); }
};

class DebugStream : public LogStream {
 public:
    explicit DebugStream(Logger& logger, bool condition = true)
        : LogStream(logger, condition && Log::debug_level_ >= DebugLevelType::FULL) {}
    DebugStream(DebugStream&& orig) = default;

    ALWAYS_INLINE ~DebugStream() {
        if (UNLIKELY(enabled_)) {
            logger_.debug("%s\n", os_.str().c_str());
        }
    }
};

class TempStream : public LogStream {
 public:
    explicit TempStream(Logger& logger) : LogStream(logger, true) {}
    TempStream(TempStream&& orig) = default;

    ~TempStream() { logger_.temp("%s\n", os_.str().c_str()); }
};

#define DEFINE_LOGGER(tag, tagString)                                                           \
    namespace Log {                                                                             \
    class tag {                                                                                 \
     public:                                                                                    \
        static ConditionalErrStream E_IF(bool exit_condition) {                                 \
            return ConditionalErrStream(getLogger(), exit_condition);                           \
        }                                                                                       \
        static DebugStream D_IF(bool condition) { return DebugStream(getLogger(), condition); } \
        static ErrStream   E() { return ErrStream(getLogger()); }                               \
        static InfoStream  I() { return InfoStream(getLogger()); }                              \
        static DebugStream D() { return DebugStream(getLogger()); }                             \
        static TempStream  T() { return TempStream(getLogger()); }                              \
        static Logger&     getLogger() {                                                        \
            static Logger logger(tagString " ");                                            \
            return logger;                                                                  \
        }                                                                                       \
    };                                                                                          \
    } // namespace Log

DEFINE_LOGGER(COMMON, "[COMMON]")
