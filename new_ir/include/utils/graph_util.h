#pragma once

#include "common/include/log.hpp"
#include "new_ir/include/types.h"
#include "new_ir/include/utils/graph_search.h"

#include "new_ir/include/utils/graph_search.h"

namespace nn_compiler {
namespace ir {

bool isSingleValueType(DataType data_type);

int32_t inferBitwidth(DataType type);

std::string ConvertDataType(const DataType previous_type);

DataType inferDataType(int32_t bitwidth, std::string data_type);

}  // namespace ir
}  // namespace nn_compiler
