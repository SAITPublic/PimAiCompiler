#pragma once

#include <torch/script.h>
#include "new_ir/include/layers/all_layers.h"
#include "new_ir/include/nn_model.h"
#include "new_runtime/include/executor/aten_ops.h"
#include "new_runtime/include/executor/stream_executor.h"
#include "new_runtime/include/executor/utils.h"

namespace nn_compiler
{
namespace runtime
{

void executorAtenAdd(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenAddmm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenAnd(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenAny(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenAppend(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenArange1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenArange2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenArange3(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenAsTensor(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenBatchNorm2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenBitwiseNot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenBmm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenBool(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenCat(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenCeil(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenChunk(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenClamp(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenClear(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenContiguous(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenConv2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenCopy(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenCpu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenCuda(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenDeriveIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenDim(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenDiv(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenDropout(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenEmbedding(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenEq(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenEqual(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenExpand(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenFill(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenFloorDivide(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenFormat(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenGather(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenGe(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenGetItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenGt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenGetItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenIndex(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenIndexPut(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenIndexSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenInt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenIs(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLeakyRelu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLen(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLinear(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenList(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLog(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLogSoftmax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLSTM1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLSTM2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenLt(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenMaskedFill(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenMaskedSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenMatmul(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenMax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenMaxPool2d(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenMin(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenMul(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenNe(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenNeg(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenNorm(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenNot(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenOnes(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenPackPaddedSequence(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenPadPackedSequence(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenPow(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenRelu(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenReshape(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSelect(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSetItem(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSize(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSlice(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSoftmax(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSqueeze(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSub(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenSum(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenTanh(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenTensor(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenTo1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenTo2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenTopk(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenTranspose(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenUnsqueeze(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenView(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenWarn(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenZeros(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
void executorAtenZerosLike(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor);
std::vector<int64_t> getDataShapeFromVector(const std::vector<int64_t>& value);
}  // namespace runtime
}  // namespace nn_compiler
