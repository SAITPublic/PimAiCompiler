#pragma once

#include <torch/script.h>
#include "executor/prim_ops.h"
#include "executor/stream_executor.h"
#include "glog/logging.h"
#include "ir/include/all_nodes.hpp"
#include "ir/include/nn_ir.hpp"

namespace nnrt
{
void executePrimBlock(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimConstant(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimData(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimDevice(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimDtype(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimEndLoop(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimEndLoop(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimEndIf(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimIf(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimLoop(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimLoopIndex(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimListConstruct(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimListUnpack(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimRaiseException(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimTupleConstruct(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimTupleIndex(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimTupleUnpack(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimUncheckedCast(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimUninitialized(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executePrimVariable(const nncir::Node& op_node, StreamExecutor& stream_executor);
}  // namespace nnrt
