#pragma once

#include <torch/script.h>
#include "executor/aten_ops.h"
#include "executor/stream_executor.h"
#include "glog/logging.h"
#include "ir/include/nn_ir.hpp"

// class ExecutorContext;

namespace nnrt
{
void executorAtenAdd(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAddmm(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAnd(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAny(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAppend(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenArange1(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenArange2(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenArange3(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenAsTensor(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenBitwiseNot(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenBmm(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenBool(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenCat(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenCeil(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenChunk(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenClamp(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenClear(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenContiguous(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenConv2d(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenCopy(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenCpu(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenCuda(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenDeriveIndex(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenDim(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenDiv(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenDropout(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenEmbedding(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenEq(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenEqual(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenExpand(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenFill(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenFloorDivide(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenFormat(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenGather(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenGe(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenGetItem(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenGt(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenGetItem(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenIndex(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenIndexPut(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenIndexSelect(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenInt(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenIs(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenItem(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLeakyRelu(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLen(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLinear(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenList(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLog(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLogSoftmax(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLSTM(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenLt(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenMaskedFill(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenMatmul(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenMax(const nncir::Node& op_node, StreamExecutor& stream_executor);
void executorAtenMaxPool2d(const nncir::Node& op_node, StreamExecutor& stream_executor);
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
