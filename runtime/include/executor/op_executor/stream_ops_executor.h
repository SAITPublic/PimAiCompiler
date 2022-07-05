#pragma once

#include <torch/script.h>
#include "executor/stream_executor.h"

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
void executeStartMultiStream(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executeEndMultiStream(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
