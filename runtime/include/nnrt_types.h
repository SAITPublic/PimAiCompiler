#pragma once

#include "common/include/types.hpp"
#include "osal_types.h"

namespace nnrt
{

// these dtype are same with GraphGen
enum DataType {
    UNDEFINED = 0,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    INT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    BOOL,
    STRING,
    DEVICE,
    TENSOR,
    NONE,
    LIST
};

struct OpNodeDescription {
    int id;
    std::string type;

    OpNodeDescription() {}
    OpNodeDescription(int id, std::string type) : id(id), type(type) {}
};

}  // namespace nnrt
