#pragma once

#include "executor/stream_executor.h"
#include "executor/utils/utils.h"
#include "ir/include/common/utils.hpp"
#include "utils/utils.h"

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
void executeMIOpenLSTM1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executeMIOpenLSTM2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
