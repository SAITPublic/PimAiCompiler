#pragma once

#include <torch/script.h>
#include "new_runtime/include/executor/aten_ops.h"
#include "new_runtime/include/executor/stream_executor.h"
#include "new_runtime/include/executor/utils.h"

// class ExecutorContext;

namespace nn_compiler
{
namespace runtime
{
void executorAtenReshape(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);

}  // namespace runtime
}  // namespace nn_compiler
