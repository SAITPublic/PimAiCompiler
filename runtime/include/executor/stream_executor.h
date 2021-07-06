#pragma once

#include <torch/script.h>
#include <functional>
#include <vector>
#include <string>
#include "ir/include/nn_ir.hpp"
#include "model_builder.h"
#include "nnrt_types.h"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/nn_ir.hpp"

// #include "prim_ops_executor.h"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{

class StreamExecutor;
using OpExecutorFn = std::function<void(const nncir::Node&, StreamExecutor& stream_executor)>;

class StreamExecutor
{
   public:

    StreamExecutor(const std::shared_ptr<nncir::NNIR> ir_graph_) { 
        registerOp();

        // Get the output & input node from ir_graph at once
        this->input_blob_ids_.clear();
        this->output_blob_ids_.clear();
        for(auto& op_node : ir_graph_->getNodes()){
            if(op_node.getNodeType() == nncir::NodeType::PRIMINPUT){
                auto& data_edge = cast<nncir::DataEdge>(op_node.getOutEdge(0));
                this->input_blob_ids_.push_back(data_edge.getBlobId());
            }else if(op_node.getNodeType() == nncir::NodeType::PRIMOUTPUT){
                auto& data_edge = cast<nncir::DataEdge>(op_node.getInEdge(0));
                this->output_blob_ids_.push_back(data_edge.getBlobId());
            }
        }

        DLOG(INFO) << "Num inputs of Graph:" << this->input_blob_ids_.size();
        DLOG(INFO) << "Num outputs of Graph:" << this->output_blob_ids_.size();

        if (this->input_blob_ids_.size() == 0 || this->output_blob_ids_.size() == 0) {
            DLOG(ERROR) << "The Graph must have >= 1 inputs and outputs!";
        }

    }

    
    RetVal inferenceModel(const std::shared_ptr<nncir::NNIR> runnable_ir,
                          const std::vector<torch::Tensor>& input_tensors, std::vector<torch::Tensor>& output_tensors);

    void updateBlob(int64_t blob_id, DataType dtype, const torch::jit::IValue& iv);

    std::pair<DataType, torch::jit::IValue>& findBlob(int64_t blob_id);

    OpExecutorFn findOpExecutor(nncir::NodeType op_type);

    void registerOp();

    void setInputTensors(const std::vector<torch::Tensor>& input_tensors);

    void getOutputTensors(std::vector<torch::Tensor>& output_tensors);

   public:
    // Global input & output vars
    std::unordered_map<int64_t, std::pair<DataType, torch::jit::IValue>> global_blobs_;
    // Op Register
    std::unordered_map<nncir::NodeType, OpExecutorFn> global_op_register_;

    std::vector<int64_t> input_blob_ids_;
    std::vector<int64_t> output_blob_ids_;
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

