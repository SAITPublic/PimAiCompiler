#pragma once

#include <torch/script.h>
#include <functional>
#include <stack>
#include <string>
#include <vector>
#include <miopen/miopen.h>
#include "builder/model_builder.h"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/nn_ir.hpp"
#include "nnrt_types.h"
#include "runtime/include/profiler.h"
#include "utils.h"

// #include "prim_ops_executor.h"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
class StreamExecutor;
using OpExecutorFn = std::function<void(const nncir::Node&, StreamExecutor& stream_executor)>;

class StreamExecutor
{
   public:
    StreamExecutor(const std::shared_ptr<nncir::NNIR> ir_graph, std::string model_type = "");
    ~StreamExecutor();

    void loadWeightAndBias(nncir::Blob* blob);

    RetVal inferenceModel(const std::shared_ptr<nncir::NNIR> runnable_ir,
                          const std::vector<torch::Tensor>& input_tensors,
                          std::vector<torch::Tensor>& output_tensors);

    RetVal inferenceModelwithProfiling(const std::shared_ptr<nncir::NNIR> runnable_ir,
                                       const std::vector<torch::Tensor>& input_tensors,
                                       std::vector<torch::Tensor>& output_tensors);

    void updateBlob(int64_t blob_id, DataType dtype, const torch::jit::IValue& iv);

    std::pair<DataType, torch::jit::IValue>& findBlob(int64_t blob_id);

    OpExecutorFn findOpExecutor(nncir::NodeType op_type);

    void registerOp();

    void setInputTensors(const std::vector<torch::Tensor>& input_tensors);

    void getOutputTensors(std::vector<torch::Tensor>& output_tensors);
    
    std::vector<torch::Tensor> iValueParser(torch::jit::IValue &iv);

    const std::shared_ptr<nncir::NNIR> getGraph();

    void setCursor(int64_t cursor)
    {
        assert(cursor >= 0);
        cursor_ = cursor;
    }

    void showAllBlobs()
    {
        for (auto& item : this->global_blobs_) {
            DLOG(INFO) << "blob_id: " << item.first << " dtype: " << getDataTypeStr(item.second.first);
        }
    }

    RetVal showBlob(int64_t blob_id)
    {
        auto it = this->global_blobs_.find(blob_id);
        if (it == this->global_blobs_.end()) {
            DLOG(INFO) << "Blob not found!";
            return RetVal::FAILURE;
        } else {
            DLOG(INFO) << "blob_id: " << blob_id << " type:" << getDataTypeStr(it->second.first)
                       << " value: " << it->second;
            return RetVal::SUCCESS;
        }
    }

   public:
    // Global input & output vars
    std::unordered_map<int64_t, std::pair<DataType, torch::jit::IValue>> global_blobs_;
    // Op Register
    std::unordered_map<nncir::NodeType, OpExecutorFn> global_op_register_;
    std::vector<int64_t> input_blob_ids_;
    std::vector<int64_t> output_blob_ids_;
    std::shared_ptr<nncir::NNIR> ir_graph_;
    int64_t cursor_ = 0;  // like the program counter
    std::stack<int64_t> loop_condition_stack_;
    std::unordered_map<int, std::pair<int, int>> releation_blob_ids_map_;

    // miopen
    miopenTensorDescriptor_t input_tensor, hidden_tensor, weight_tensor, output_tensor;
    std::vector<miopenTensorDescriptor_t> input_tensors;
    std::vector<miopenTensorDescriptor_t> output_tensors;
    miopenHandle_t handle;
    miopenRNNDescriptor_t rnnDesc;

    std::string modelType;
};

}  // namespace nnrt
