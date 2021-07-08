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

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
// RetVal StreamExecutor::inferenceModel(const std::shared_ptr<nncir::NNIR> graph,
//                                       const std::vector<torch::Tensor>& input_tensors,
//                                       std::vector<torch::Tensor>& output_tensors)
// {
//     // Set input tensors
//     int idx = 0;
//     for (auto& tensor : input_tensors) {
//         this->updateBlob(idx++, DataType::TENSOR, tensorToIValue(tensor));
//     }

//     // Execute Graph
//     for (auto& node : graph->getNodes()) {
//         LOG(INFO) << "Node id:" << node.getId() << " name:" << node.getName() << " type:" << node.getNodeType();
//         auto node_type = node.getNodeType();

//         if (node_type == nncir::NodeType::PRIMIF) {
//             // Get initial cond
//             // PrimIf only have one input, convert to DataEdge
//             auto& data_edge = cast<nncir::DataEdge>(node.getFirstInEdge());

//             int64_t blob_id = data_edge.getBlobId();
//             assert(findBlob(blob_id).first == DataType::INT64);
//             auto cond = findBlob(blob_id).second.toInt()

//             if (cond) {
//                 // Get first output edge --> node
//                 // execute to EndIf Node
//                 auto true_edge = node.getOutEdge(0);
//                 node = true_edge.dst;

//             } else {
//                 auto op_node = node.getOutEdge(1);
//             }

//             while(op_node.getNodeType()!=nncir::NodeType::PRIMENDIF){

//             }

//             // TODO
//         } else if (node_type == nncir::NodeType::PRIMLOOP) {
//             // TODO
//         } else {
//             // Call execute Op
//             auto op_executor = this->findOpExecutor(node.getNodeType());
//             op_executor(node, *this);
//         }
//     }

//     return RetVal::SUCCESS;
// }


RetVal StreamExecutor::inferenceModel(const std::shared_ptr<nncir::NNIR> graph,
                                      const std::vector<torch::Tensor>& input_tensors,
                                      std::vector<torch::Tensor>& output_tensors)
{
    // Set input tensors
    int idx = 0;
    for (auto& tensor : input_tensors) {
        this->updateBlob(idx++, DataType::TENSOR, tensorToIValue(tensor));
    }

    // Execute Graph
    int cursor = 0;
    for(cursor = 0; cursor < graph->getNodeCount(); ) {

        auto op_node = graph->getNode(cursor);
        auto node_type = op_node->getNodeType();

        LOG(INFO) << "Node id:" << op_node->getId() << " name:" << op_node->getName() << " type:" << op_node->getNodeType();

        if(node_type == nncir::NodeType::PRIMIF){
            // Get initial cond
            // PrimIf only have one input, convert to DataEdge
            auto& data_edge = cast<nncir::DataEdge>(op_node->getFirstInEdge());
            int64_t blob_id = data_edge.getBlobId();
            assert(findBlob(blob_id).first == DataType::INT64);
            auto cond = findBlob(blob_id).second.toInt();

            // In GraphIR, the true branch is firstOutput Edge, the false branch
            // is SecondOutput Edge
            if(cond){
                auto& edge_true =  cast<nncir::DataEdge>(op_node->getOutEdge(0));
                // move sursor
                cursor = edge_true.getOutNodeId();
            }else{
                auto& edge_false =  cast<nncir::DataEdge>(op_node->getOutEdge(1));
                cursor = edge_false.getOutNodeId();
            }
        } else if (node_type == nncir::NodeType::PRIMLOOP) {
            // TODO

        } else {

            // Call execute Op
            auto op_executor = this->findOpExecutor(op_node->getNodeType());
            op_executor(*op_node, *this);
            cursor++;
        }

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
    assert(it != this->global_op_register_.end());
    return it->second;
}

void StreamExecutor::registerOp()
{
    // Register Prim Ops: {OP_TYPE, OP_FUNCTION}
    this->global_op_register_.insert({nncir::NodeType::ATENADD, executorAtenAdd});
    this->global_op_register_.insert({nncir::NodeType::PRIMCONSTANT, executePrimConstant});
    this->global_op_register_.insert({nncir::NodeType::PRIMDTYPE, executePrimDtype});
    // Register Aten Ops
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
