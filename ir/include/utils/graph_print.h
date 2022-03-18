#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/nn_model.h"

namespace nn_compiler
{
namespace ir
{
void printGraphModel(std::unique_ptr<ir::NNModel>& nn_model);

}  // namespace ir
}  // namespace nn_compiler
