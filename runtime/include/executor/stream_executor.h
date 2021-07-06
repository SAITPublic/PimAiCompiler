#pragma once

#include <torch/script.h>
#include <functional>
#include <string>
#include "ir/include/nn_ir.hpp"
#include "model_builder.h"
#include "nnrt_types.h"
// #include "prim_ops_executor.h"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
class StreamExecutor;
typedef void (*OpExecutorFn)(const nncir::Node&, StreamExecutor& stream_executor);

class StreamExecutor
{
   public:
    StreamExecutor() { registerOp(); }

    RetVal inferenceModel(const std::shared_ptr<nncir::NNIR> runnable_ir,
                          const std::vector<torch::Tensor>& input_tensors, std::vector<torch::Tensor>& output_tensors);

    void updateBlob(int64_t blob_id, DataType dtype, const torch::jit::IValue& iv);

    std::pair<DataType, torch::jit::IValue>& findBlob(int64_t blob_id);

    OpExecutorFn findOpExecutor(nncir::NodeType op_type);

    void registerOp();

   public:
    // Global input & output vars
    std::unordered_map<int64_t, std::pair<DataType, torch::jit::IValue>> global_blobs_;
    // Op Register
    std::unordered_map<nncir::NodeType, OpExecutorFn> global_op_register_;
};

// execute current op in runtime
void executeOp(OpNodeDescription* cur_op);

/**
 * @brief Get the Next Execution Node object
 *
 * @param cur_op currently Op
 * @return OpNodeDescription* the next
 */
OpNodeDescription* getNextExecutionOp(OpNodeDescription* cur_op);

}  // namespace nnrt

