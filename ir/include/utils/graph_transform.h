#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/nn_model.h"

namespace nn_compiler
{
namespace ir
{
namespace utils
{
void deleteLayer(std::shared_ptr<ir::NNNetwork> graph, std::shared_ptr<ir::NNLayer> layer);

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
