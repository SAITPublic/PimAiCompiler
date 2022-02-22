#pragma once

#include "common/include/log.hpp"
#include "new_ir/include/types.h"

namespace nn_compiler {
namespace ir {

bool isSingleValueType(DataType data_type);

int32_t inferBitwidth(DataType graphgen_type);

std::string ConvertDataType(const DataType previous_type);

DataType inferDataType(int32_t bitwidth, std::string data_type);

}  // namespace ir
}  // namespace nn_compiler
