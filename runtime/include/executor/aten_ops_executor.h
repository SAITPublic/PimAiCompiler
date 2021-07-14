#pragma once

#include <torch/script.h>
#include "executor/aten_ops.h"
#include "executor/stream_executor.h"
#include "glog/logging.h"
#include "ir/include/nn_ir.hpp"

// class ExecutorContext;

namespace nnrt
{
void executorAtenDeriveIndex(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenGetItem(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenIs(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAdd(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAddMM(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAppend(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenCat(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenCeil(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenDim(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenDiv(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenDropout(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenEmbedding(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenEq(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenFormat(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenGt(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenInt(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenItem(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLen(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenList(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLt(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenMatmul(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenMax(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenNe(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenNeg(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenRelu(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenSelect(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenSize(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenSlice(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenSub(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenTensor(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenTo(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenTranspose(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenUnsqueeze(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenZeros(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenZerosLike(const nncir::Node& op_node, StreamExecutor& stream_executor);

}  // namespace nnrt
