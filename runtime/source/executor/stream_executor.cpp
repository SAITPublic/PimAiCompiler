#include "executor/stream_executor.h"
#include <torch/script.h>
#include "common/include/cast.hpp"
#include "executor/aten_ops_executor.h"
#include "executor/prim_ops_executor.h"
#include "executor/utils.h"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"
#include "nnrt_types.h"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
RetVal StreamExecutor::inferenceModel(const std::shared_ptr<nncir::NNIR> graph,
                                      const std::vector<torch::Tensor>& input_tensors,
                                      std::vector<torch::Tensor>& output_tensors)
{
    // Set Input Tensors
    for (auto& in : input_tensors) {
        DLOG(INFO) << "Input Tensor:" << in.sizes() << " data:" << in;
    }
    this->setInputTensors(input_tensors);

    // cursor skiped the InputNode and OutputNode
    // [cursor_begin, cursor_end)
    int cursor_begin = input_tensors.size();
    int cursor_end = graph->getNodeCount() - this->output_blob_ids_.size();

    // control_op will move cursor by itself
    auto is_control_op = [](nncir::NodeType type) {
        return (type == nncir::NodeType::PRIMIF || type == nncir::NodeType::PRIMENDIF ||
                type == nncir::NodeType::PRIMLOOP || type == nncir::NodeType::PRIMENDLOOP ||
                type == nncir::NodeType::PRIMBLOCK);
    };

    // Execute Graph
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        nncir::Node* node = graph->getNode(cursor_);
        DLOG(INFO) << "Node id:" << node->getId() << " name:" << node->getName() << " type:" << node->getNodeType();
        auto node_type = node->getNodeType();
        auto op_executor = this->findOpExecutor(node_type);
        op_executor(*node, *this);

        if (!is_control_op(node_type)) {
            cursor_++;
        }
    }

    // Read Output Tensors
    this->getOutputTensors(output_tensors);
    for (auto& out : output_tensors) {
        DLOG(INFO) << "Output Tensor:" << out.sizes() << " data:" << out;
    }

    // for debug
    // for debug
    this->showAllBlobs();
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
    if (it == this->global_op_register_.end()) {
        DLOG(ERROR) << "Runtime error, Unregistered Op !";
    }
    assert(it != this->global_op_register_.end());
    return it->second;
}

void StreamExecutor::registerOp()
{
    this->global_op_register_.insert({nncir::NodeType::ATENADD, executorAtenAdd});
    this->global_op_register_.insert({nncir::NodeType::ATENADDMM, executorAtenAddmm});
    this->global_op_register_.insert({nncir::NodeType::ATENAPPEND, executorAtenAppend});
    this->global_op_register_.insert({nncir::NodeType::ATENCAT, executorAtenCat});
    this->global_op_register_.insert({nncir::NodeType::ATENCEIL, executorAtenCeil});
    this->global_op_register_.insert({nncir::NodeType::ATENCOPY, executorAtenCopy});
    this->global_op_register_.insert({nncir::NodeType::ATENDERIVEINDEX, executorAtenDeriveIndex});
    this->global_op_register_.insert({nncir::NodeType::ATENDIM, executorAtenDim});
    this->global_op_register_.insert({nncir::NodeType::ATENDIV, executorAtenDiv});
    this->global_op_register_.insert({nncir::NodeType::ATENDROPOUT, executorAtenDropout});
    this->global_op_register_.insert({nncir::NodeType::ATENEMBEDDING, executorAtenEmbedding});
    this->global_op_register_.insert({nncir::NodeType::ATENEQ, executorAtenEq});
    this->global_op_register_.insert({nncir::NodeType::ATENEXPAND, executorAtenExpand});
    this->global_op_register_.insert({nncir::NodeType::ATENFORMAT, executorAtenFormat});
    this->global_op_register_.insert({nncir::NodeType::ATENGETITEM, executorAtenGetItem});
    this->global_op_register_.insert({nncir::NodeType::ATENGT, executorAtenGt});
    this->global_op_register_.insert({nncir::NodeType::ATENINT, executorAtenInt});
    this->global_op_register_.insert({nncir::NodeType::ATENITEM, executorAtenItem});
    this->global_op_register_.insert({nncir::NodeType::ATENIS, executorAtenIs});
    this->global_op_register_.insert({nncir::NodeType::ATENLEN, executorAtenLen});
    this->global_op_register_.insert({nncir::NodeType::ATENLIST, executorAtenList});
    this->global_op_register_.insert({nncir::NodeType::ATENLSTM, executorAtenLSTM});
    this->global_op_register_.insert({nncir::NodeType::ATENLT, executorAtenLt});
    this->global_op_register_.insert({nncir::NodeType::ATENMATMUL, executorAtenMatmul});
    this->global_op_register_.insert({nncir::NodeType::ATENMAX, executorAtenMax});
    this->global_op_register_.insert({nncir::NodeType::ATENNE, executorAtenNe});
    this->global_op_register_.insert({nncir::NodeType::ATENNEG, executorAtenNeg});
    this->global_op_register_.insert({nncir::NodeType::ATENRELU, executorAtenRelu});
    this->global_op_register_.insert({nncir::NodeType::ATENSELECT, executorAtenSelect});
    this->global_op_register_.insert({nncir::NodeType::ATENSIZE, executorAtenSize});
    this->global_op_register_.insert({nncir::NodeType::ATENSLICE, executorAtenSlice});
    this->global_op_register_.insert({nncir::NodeType::ATENSUB, executorAtenSub});
    this->global_op_register_.insert({nncir::NodeType::ATENTENSOR, executorAtenTensor});
    this->global_op_register_.insert({nncir::NodeType::ATENTRANSPOSE, executorAtenTranspose});
    this->global_op_register_.insert({nncir::NodeType::ATENTO, executorAtenTo});
    this->global_op_register_.insert({nncir::NodeType::ATENUNSQUEEZE, executorAtenUnsqueeze});
    this->global_op_register_.insert({nncir::NodeType::ATENZEROS, executorAtenZeros});
    this->global_op_register_.insert({nncir::NodeType::ATENZEROSLIKE, executorAtenZerosLike});

    this->global_op_register_.insert({nncir::NodeType::PRIMBLOCK, executePrimBlock});
    this->global_op_register_.insert({nncir::NodeType::PRIMCONSTANT, executePrimConstant});
    this->global_op_register_.insert({nncir::NodeType::PRIMDATA, executePrimData});
    this->global_op_register_.insert({nncir::NodeType::PRIMDEVICE, executePrimDevice});
    this->global_op_register_.insert({nncir::NodeType::PRIMDTYPE, executePrimDtype});
    this->global_op_register_.insert({nncir::NodeType::PRIMENDLOOP, executePrimEndLoop});
    this->global_op_register_.insert({nncir::NodeType::PRIMIF, executePrimIf});
    this->global_op_register_.insert({nncir::NodeType::PRIMENDIF, executePrimEndIf});
    this->global_op_register_.insert({nncir::NodeType::PRIMLOOP, executePrimLoop});
    this->global_op_register_.insert({nncir::NodeType::PRIMLOOPINDEX, executePrimLoopIndex});
    this->global_op_register_.insert({nncir::NodeType::PRIMLISTCONSTRUCT, executePrimListConstruct});
    this->global_op_register_.insert({nncir::NodeType::PRIMLISTUNPACK, executePrimListUnpack});
    this->global_op_register_.insert({nncir::NodeType::PRIMRAISEEXCEPTION, executePrimRaiseException});
    this->global_op_register_.insert({nncir::NodeType::PRIMTUPLECONSTRUCT, executePrimTupleConstruct});
    this->global_op_register_.insert({nncir::NodeType::PRIMTUPLEINDEX, executePrimTupleIndex});
    this->global_op_register_.insert({nncir::NodeType::PRIMTUPLEUNPACK, executePrimTupleUnpack});
    this->global_op_register_.insert({nncir::NodeType::PRIMUNCHECKEDCAST, executePrimUncheckedCast});
    this->global_op_register_.insert({nncir::NodeType::PRIMUNINITIALIZED, executePrimUninitialized});
    this->global_op_register_.insert({nncir::NodeType::PRIMVARIABLE, executePrimVariable});
}

void StreamExecutor::setInputTensors(const std::vector<torch::Tensor>& input_tensors)
{
    if (input_tensors.size() != this->input_blob_ids_.size()) {
        DLOG(ERROR) << "Num tensors must match the num inputs of Graph,"
                    << "the Graph needs " << this->input_blob_ids_.size() << "inputs !";
    }
    // Set the input tensors to placeholder, assume all inputs & outputs are Tensor type
    int k = 0;
    for (auto& id_ : this->input_blob_ids_) {
        this->updateBlob(id_, DataType::TENSOR, tensorToIValue(input_tensors.at(k)));
        k++;
    }
}

void StreamExecutor::getOutputTensors(std::vector<torch::Tensor>& output_tensors)
{
    output_tensors.clear();
    // Read the output tensors
    for (auto& id_ : this->output_blob_ids_) {
        auto blob = this->findBlob(id_);
        output_tensors.push_back(blob.second.toTensor());
    }
}

const std::shared_ptr<nncir::NNIR> StreamExecutor::getGraph() { return this->ir_graph_; }

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
