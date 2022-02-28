#pragma once

#include <functional>
#include <miopen/miopen.h>
#include <stack>
#include <string>
#include <torch/script.h>
#include <vector>

#include "builder/model_builder.h"
#include "new_ir/include/nn_model.h"
#include "new_runtime/include/types.h"
#include "new_runtime/include/executor/utils.h"

namespace nn_compiler {
namespace runtime {

class StreamExecutor
{
   public:
    typedef std::unordered_map<int64_t, std::pair<DataType, torch::jit::IValue>> blob_store_type;

    StreamExecutor(std::pair<std::unique_ptr<nn_compiler::ir::NNModel>, blob_store_type> model, std::string model_type);

    ~StreamExecutor();

    RetVal inferenceModel(const std::vector<torch::Tensor>& input_tensors,
                          std::vector<torch::Tensor>& output_tensors);

    RetVal inferenceModelwithProfiling(const std::vector<torch::Tensor>& input_tensors,
                                       std::vector<torch::Tensor>& output_tensors);

    void updateBlob(int64_t blob_id, DataType dtype, const torch::jit::IValue& iv);

    std::pair<DataType, torch::jit::IValue>& findBlob(int64_t blob_id);

    // OpExecutorFn findOpExecutor(node_type);

    void registerOp();

    void setInputTensors(const std::vector<torch::Tensor>& input_tensors);

    void getOutputTensors(std::vector<torch::Tensor>& output_tensors);
    
    std::vector<torch::Tensor> iValueParser(torch::jit::IValue &iv);

    void setCursor(int64_t cursor)
    {
        assert(cursor >= 0);
        cursor_ = cursor;
    }

   public:
    std::unique_ptr<nn_compiler::ir::NNModel> model_ = nullptr;

    // Global input & output vars
    blob_store_type global_blobs_;
    // Op Register
    //std::unordered_map<nncir::NodeType, OpExecutorFn> global_op_register_;

    std::vector<int64_t> input_blob_ids_;
    std::vector<int64_t> output_blob_ids_;

    int64_t cursor_ = 0;  // like the program counter

    std::stack<int64_t> loop_condition_stack_;
    std::unordered_map<int, std::pair<int, int>> releation_blob_ids_map_;
};

}  // namespace runtime
}  // namespace nn_compiler
