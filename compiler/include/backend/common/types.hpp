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
 * @file.    types.hpp
 * @brief.   this file define primitive types.
 * @details. This header defines primitive types for NN compiler
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"

enum class RetVal {
    SUCCESS = 0,
    FAILURE = 1,
    // need to add more errer return types
};

enum class DebugLevelType {
    NONE    = 0, // do not print any log
    PARTIAL = 1, // print logs only about call sequence
    FULL    = 2, // print all logs (call sequence + debug info)
};

enum class BackEndOutputType {
    CSV  = 0, // only CSV out
    BIN  = 1, // only bin out
    BOTH = 2, // CSV, bin out
    NONE = 3, //
};

enum class Target {
    EVT0 = 0,
    EVT1 = 1,
};

enum class RunEnv { SYSTEMC = 0, SILICON = 1 };

enum class CompilerType { ONDEVICE = 0, OFFLINE = 1 };

/// @brief user-defined literal to get kilobytes (actually kibibytes) by bytes
constexpr auto operator""_KB(unsigned long long bytes) { return bytes << 10; }

/// @brief user-defined literal to get megabytes (actually mebibytes) by bytes
constexpr auto operator""_MB(unsigned long long bytes) { return bytes << 20; }
