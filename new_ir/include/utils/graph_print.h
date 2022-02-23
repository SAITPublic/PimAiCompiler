#pragma once

#include "new_ir/include/common/log.hpp"
#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/nn_model.h"

namespace nn_compiler {
namespace ir {

void printGraphModel(std::unique_ptr<ir::NNModel>& nn_model);

} // namespace ir
} // namespace nn_compiler
