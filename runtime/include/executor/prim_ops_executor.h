#pragma once

#include <torch/script.h>
#include "executor/prim_ops.h"
#include "executor/stream_executor.h"
#include "glog/logging.h"
#include "ir/include/nn_ir.hpp"

// class ExecutorContext;

namespace nnrt
{
void executePrimConstant(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimDtype(const nncir::Node& op_node, StreamExecutor& stream_executor);

}  // namespace nnrt
