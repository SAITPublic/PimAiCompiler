#pragma once

#include "ir/include/nn_model.h"

namespace nn_compiler
{
namespace ir
{
namespace utils
{
void deleteLayer(std::shared_ptr<ir::NNGraph> graph, std::shared_ptr<ir::NNLayer> layer);

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
