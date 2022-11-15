/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#pragma once

#include <float.h>
#include <typeinfo>

namespace nn_compiler::ir
{
inline bool isDefaultValue(int& input) { return input == INT32_MAX; }
inline bool isDefaultValue(int64_t& input) { return input == INT64_MIN; }
inline bool isDefaultValue(float& input) { return (fabs(input - FLT_MAX) <= FLT_EPSILON); }
inline bool isDefaultValue(double& input) { return (fabs(input - DBL_MAX) <= DBL_EPSILON); }
inline bool isDefaultValue(std::string& input) { return input == ""; }
}  // namespace nn_compiler::ir
