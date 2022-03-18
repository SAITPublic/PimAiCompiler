#pragma once

#include "ir/include/types.h"
#include "ir/include/utils/graph_print.h"
#include "ir/include/utils/graph_search.h"
#include "ir/include/utils/graph_transform.h"

namespace nn_compiler
{
namespace ir
{
bool isSingleValueType(DataType data_type);

int32_t inferBitwidth(DataType type);

std::string ConvertDataType(const DataType previous_type);

DataType inferDataType(int32_t bitwidth, std::string data_type);

}  // namespace ir
}  // namespace nn_compiler
