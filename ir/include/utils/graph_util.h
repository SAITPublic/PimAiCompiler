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

#include "ir/include/utils/graph_print.h"
#include "ir/include/utils/graph_search.h"

namespace nn_compiler
{
namespace ir
{
namespace utils
{
bool isSingleValueType(DataType data_type);

int32_t inferBitwidth(DataType type);

std::string ConvertDataType(const DataType previous_type);

DataType inferDataType(int32_t bitwidth, std::string data_type);

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
