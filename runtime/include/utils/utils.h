/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#pragma once

#include <torch/script.h>

#include "executor/utils/utils.h"
#include "ir/include/tensors/data_tensor.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace runtime
{
namespace utils
{
using namespace nn_compiler::ir;

torch::Tensor createPtTensor(void* data_ptr, const std::vector<int64_t>& shape, DataType dtype,
                             const std::vector<int64_t>& stride = {});

std::vector<int64_t> getDataShapeFromSTensor(nn_compiler::ir::STensor& stensor);

torch::Tensor loadTensor(const std::string& bin_file, const std::vector<int64_t>& shape, DataType dtype);

std::pair<int, DataType> parseNtype(std::string& ntype);

torch::jit::IValue convertVaraibleData2IValve(uint8_t* ptr, DataType d_type);

}  // namespace utils
}  // namespace runtime
}  // namespace nn_compiler
