#pragma once

#include "ir/include/common/log.hpp"

#include <typeinfo>

#include <float.h>

#include <math.h>

namespace nn_compiler::nn_ir {

template <typename T>
bool isDefaultValue(T& input) {
    int a;
    int64_t b;
    float c;
    double d;
    if (typeid(input).name() == typeid(a).name() && input == INT32_MAX) {
        return true;
    } else if (typeid(input).name() == typeid(b).name() && input == INT64_MAX) {
        return true;
    } else if (typeid(input).name() == typeid(c).name() && (fabs(input - FLT_MAX) <= FLT_EPSILON)) {
        return true;
    } else if (typeid(input).name() == typeid(d).name() && (fabs(input - DBL_MAX) <= DBL_EPSILON)) {
        return true;
    } else {
        return false;
    }
}
}  // namespace nn_compiler::nn_ir
