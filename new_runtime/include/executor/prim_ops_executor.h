#pragma once

#include <torch/script.h>
#include "new_runtime/include/executor/stream_executor.h"
#include "new_ir/include/layers/all_layers.h"

namespace nn_compiler
{
namespace runtime
{

void executePrimConstant(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executePrimVariable(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);

}  // namespace runtime
}  // namespace nn_compiler