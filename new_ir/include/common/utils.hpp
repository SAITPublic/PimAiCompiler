#pragma once

#include "new_ir/include/common/log.hpp"

#include <float.h>
#include <math.h>
#include <typeinfo>

namespace nn_compiler::ir
{
inline bool isDefaultValue(int& input) { return input == INT32_MAX; }
inline bool isDefaultValue(int64_t& input) { return input == INT64_MIN; }
inline bool isDefaultValue(float& input) { return (fabs(input - FLT_MAX) <= FLT_EPSILON); }
inline bool isDefaultValue(double& input) { return (fabs(input - DBL_MAX) <= DBL_EPSILON); }
inline bool isDefaultValue(std::string& input) { return input == ""; }
}  // namespace nn_compiler::ir
