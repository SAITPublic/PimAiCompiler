#include <torch/script.h>
#include "ir/include/nn_ir.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "common/include/cast.hpp"
#include "nnrt_types.h"
#include "executor/utils.h"
#include "executor/stream_executor.h"
#include "executor/aten_ops_executor.h"
#include "executor/prim_ops_executor.h"
#include "ir/include/control_nodes/prim_constant_node.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{

RetVal StreamExecutor::inferenceModel(const std::shared_ptr<nncir::NNIR> graph,
                                      const std::vector<torch::Tensor>& input_tensors,
                                      std::vector<torch::Tensor>& output_tensors)
{
    // Set Input Tensors
    for(auto& in : input_tensors) {
        LOG(INFO) << "Input Tensor:" <<in.sizes() <<" data:"<<in;
    }
    this->setInputTensors(input_tensors);

    // Execute Graph
    for (auto& node : graph->getNodes()) {
        LOG(INFO) << "Node id:" << node.getId() << " name:" << node.getName() << " type:" << node.getNodeType();
        auto node_type = node.getNodeType();


        if(node_type != nncir::NodeType::PRIMINPUT && node_type != nncir::NodeType::PRIMOUTPUT){
            // Call execute Op
            auto op_executor = this->findOpExecutor(node.getNodeType());
            op_executor(node, *this);
        }
    }

    // Read Output Tensors
    this->getOutputTensors(output_tensors);
    for(auto& out : output_tensors) {
        LOG(INFO) << "Output Tensor:" <<out.sizes() <<" data:"<<out;
    }

    return RetVal::SUCCESS;
}

void StreamExecutor::updateBlob(int64_t blob_id, DataType dtype, const torch::jit::IValue& iv)
{
    auto it = this->global_blobs_.find(blob_id);
    if (it == this->global_blobs_.end()) {
        // Not exist, insert
        this->global_blobs_.insert({blob_id, {dtype, iv}});
    } else {
        // exist
        assert(it->second.first == dtype);
        it->second.second = iv;
    }
}

std::pair<DataType, torch::jit::IValue>& StreamExecutor::findBlob(int64_t blob_id)
{
    auto it = global_blobs_.find(blob_id);
    assert(it != this->global_blobs_.end());
    return it->second;
}

OpExecutorFn StreamExecutor::findOpExecutor(nncir::NodeType op_type)
{
    auto it = this->global_op_register_.find(op_type);
    if(it == this->global_op_register_.end()) {
        DLOG(ERROR) << "Runtime error, Unregistered Op !";
    }
    assert(it != this->global_op_register_.end());
    return it->second;
}

void StreamExecutor::registerOp()
{
    // Register Ops: {OP_TYPE, OP_FUNCTION}
    this->global_op_register_.insert({nncir::NodeType::ATENADD, executorAtenAdd});
    this->global_op_register_.insert({nncir::NodeType::ATENCAT, executorAtenCat});
    this->global_op_register_.insert({nncir::NodeType::ATENDIV, executorAtenDiv});
    this->global_op_register_.insert({nncir::NodeType::ATENEQ, executorAtenEq});
    this->global_op_register_.insert({nncir::NodeType::ATENINT, executorAtenInt});
    this->global_op_register_.insert({nncir::NodeType::ATENNE, executorAtenNe});
    this->global_op_register_.insert({nncir::NodeType::ATENSELECT, executorAtenSelect});
    this->global_op_register_.insert({nncir::NodeType::ATENTRANSPOSE, executorAtenTranspose});
    this->global_op_register_.insert({nncir::NodeType::ATENTO, executorAtenTo});

    this->global_op_register_.insert({nncir::NodeType::PRIMCONSTANT, executePrimConstant});
    this->global_op_register_.insert({nncir::NodeType::PRIMDTYPE, executePrimDtype});
}

void StreamExecutor::setInputTensors(const std::vector<torch::Tensor>& input_tensors) {
    if(input_tensors.size() != this->input_blob_ids_.size()) {
        DLOG(ERROR) << "Num tensors must match the num inputs of Graph," <<"the Graph needs "<<
                    this->input_blob_ids_.size()<<"inputs !";
    }
    // Set the input tensors to placeholder, assume all inputs & outputs are Tensor type
    int k = 0;
    for(auto& id_ : this->input_blob_ids_) {
        this->updateBlob(id_, DataType::TENSOR, tensorToIValue(input_tensors.at(k)));
        k++;
    }
}

void StreamExecutor::getOutputTensors(std::vector<torch::Tensor>& output_tensors){
    output_tensors.clear();
    // Read the output tensors
    for(auto& id_ : this->output_blob_ids_) {
        auto blob = this->findBlob(id_);
        output_tensors.push_back(blob.second.toTensor());
    }
}

void executeOp(OpNodeDescription* cur_op) {}

/**
 * @brief Get the Next Execution Node object
 *
 * @param cur_op currently Op
 * @return OpNodeDescription* the next
 */
OpNodeDescription* getNextExecutionOp(OpNodeDescription* cur_op)
{
    // TODO
    return nullptr;
}

}  // namespace nnrt
