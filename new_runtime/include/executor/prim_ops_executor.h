#pragma once

#include <torch/script.h>

#include "new_ir/include/layers/all_layers.h"
#include "new_ir/include/layers/nn_layer.h"
#include "new_runtime/include/executor/prim_ops.h"
#include "new_runtime/include/executor/prim_utils.h"
#include "new_runtime/include/executor/stream_executor.h"

namespace nn_compiler
{
namespace runtime
{
void executePrimConstant(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executePrimTupleConstruct(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);

}  // namespace runtime
}  // namespace nn_compiler
