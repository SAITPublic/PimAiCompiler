#include "executor/stream_executor.h"
#include <torch/script.h>
#include "common/include/cast.hpp"
#include "executor/aten_ops_executor.h"
#include "executor/prim_ops_executor.h"
#include "executor/utils.h"
#include "ir/include/data_blob.hpp"
#include "ir/include/data_edge.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/nn_ir.hpp"
#include "nnrt_types.h"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
void StreamExecutor::loadWeightAndBias(nncir::Blob* blob, const std::string& kind, const std::string& device_type)
{
    nncir::Shape4D shape = blob->getShape();
    int64_t blob_id = blob->getId();

    auto data_blob = cast_if<nncir::DataBlob>(blob);
    if (data_blob == nullptr) {
        DLOG(ERROR) << "The blob is not weight or bias blob!";
    }

    auto bit_width = blob->getBitWidth();
    at::ScalarType scalar_type;
    torch::Tensor tensor_data;
    if (kind == "weight") {
        if (bit_width == 16) {
            auto value_vec = data_blob->getBuf<float16>();
            scalar_type = torch::kHalf;
            tensor_data = at::from_blob(value_vec.data(), {shape.h, shape.w}, scalar_type).cuda();
        } else if (bit_width == 32) {
            auto value_vec = data_blob->getBuf<float>();
            scalar_type = torch::kFloat;
            tensor_data = at::from_blob(value_vec.data(), {shape.h, shape.w}, scalar_type).cuda();
        } else {
            DLOG(ERROR) << "Weight bit witgh Error!";
        }

    } else if (kind == "bias") {
        if (bit_width == 16) {
            auto value_vec = data_blob->getBuf<float16>();
            scalar_type = torch::kHalf;
            tensor_data = at::from_blob(value_vec.data(), {shape.w}, scalar_type).cuda();
        } else if (bit_width == 32) {
            auto value_vec = data_blob->getBuf<float>();
            scalar_type = torch::kFloat;
            tensor_data = at::from_blob(value_vec.data(), {shape.w}, scalar_type).cuda();
        } else {
            DLOG(ERROR) << "Weight bit witgh Error!";
        }
    } else {
        DLOG(ERROR) << "Kind Error, blob kind is" << kind;
    }
    torch::jit::IValue iv = tensorToIValue(tensor_data);
    this->global_blobs_.insert({blob_id, {DataType::TENSOR, iv}});
}

StreamExecutor::StreamExecutor(const std::shared_ptr<nncir::NNIR> ir_graph)
{
    this->ir_graph_ = ir_graph;
    this->registerOp();

    // Get the output & input node from ir_graph at once
    this->input_blob_ids_.clear();
    this->output_blob_ids_.clear();
    for (auto& op_node : ir_graph_->getNodes()) {
        if (op_node.getNodeType() == nncir::NodeType::PRIMINPUT) {
            auto& data_edge = cast<nncir::DataEdge>(op_node.getOutEdge(0));
            this->input_blob_ids_.push_back(data_edge.getBlobId());
        } else if (op_node.getNodeType() == nncir::NodeType::PRIMOUTPUT) {
            auto& data_edge = cast<nncir::DataEdge>(op_node.getInEdge(0));
            this->output_blob_ids_.push_back(data_edge.getBlobId());
        } else if (op_node.getNodeType() == nncir::NodeType::ATENLSTM) {
            // For Ops' with weight/bias, firstly save to global_blobs_ once
            auto lstm_node = cast<nncir::AtenLSTMNode>(op_node);
            auto weight_blobs = lstm_node.getWeightBlob();
            auto bias_blobs = lstm_node.getBiasBlob();

            for (auto blob : weight_blobs) {
                this->loadWeightAndBias(blob, "weight", "gpu");
            }

            for (auto blob : bias_blobs) {
                this->loadWeightAndBias(blob, "bias", "gpu");
            }
        }
    }

    DLOG(INFO) << "Num inputs of Graph:" << this->input_blob_ids_.size();
    DLOG(INFO) << "Num outputs of Graph:" << this->output_blob_ids_.size();

    if (this->input_blob_ids_.size() == 0 || this->output_blob_ids_.size() == 0) {
        DLOG(ERROR) << "The Graph must have >= 1 inputs and outputs!";
    }
}

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
    // this->showAllBlobs();
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
        it->second.first = dtype;
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
    this->global_op_register_.insert({nncir::NodeType::ATENAND, executorAtenAnd});
    this->global_op_register_.insert({nncir::NodeType::ATENANY, executorAtenAny});
    this->global_op_register_.insert({nncir::NodeType::ATENAPPEND, executorAtenAppend});
    this->global_op_register_.insert({nncir::NodeType::ATENASTENSOR, executorAtenAsTensor});
    this->global_op_register_.insert({nncir::NodeType::ATENBITWISENOT, executorAtenBitwiseNot});
    this->global_op_register_.insert({nncir::NodeType::ATENBMM, executorAtenBmm});
    this->global_op_register_.insert({nncir::NodeType::ATENBOOL, executorAtenBool});
    this->global_op_register_.insert({nncir::NodeType::ATENCAT, executorAtenCat});
    this->global_op_register_.insert({nncir::NodeType::ATENCEIL, executorAtenCeil});
    this->global_op_register_.insert({nncir::NodeType::ATENCHUNK, executorAtenChunk});
    this->global_op_register_.insert({nncir::NodeType::ATENCLAMP, executorAtenClamp});
    this->global_op_register_.insert({nncir::NodeType::ATENCLEAR, executorAtenClear});
    this->global_op_register_.insert({nncir::NodeType::ATENCONTIGUOUS, executorAtenContiguous});
    this->global_op_register_.insert({nncir::NodeType::ATENCONV2D, executorAtenConv2d});
    this->global_op_register_.insert({nncir::NodeType::ATENCOPY, executorAtenCopy});
    this->global_op_register_.insert({nncir::NodeType::ATENCPU, executorAtenCpu});
    this->global_op_register_.insert({nncir::NodeType::ATENCUDA, executorAtenCuda});
    this->global_op_register_.insert({nncir::NodeType::ATENDERIVEINDEX, executorAtenDeriveIndex});
    this->global_op_register_.insert({nncir::NodeType::ATENDIM, executorAtenDim});
    this->global_op_register_.insert({nncir::NodeType::ATENDIV, executorAtenDiv});
    this->global_op_register_.insert({nncir::NodeType::ATENDROPOUT, executorAtenDropout});
    this->global_op_register_.insert({nncir::NodeType::ATENEMBEDDING, executorAtenEmbedding});
    this->global_op_register_.insert({nncir::NodeType::ATENEQ, executorAtenEq});
    this->global_op_register_.insert({nncir::NodeType::ATENEQUAL, executorAtenEqual});
    this->global_op_register_.insert({nncir::NodeType::ATENEXPAND, executorAtenExpand});
    this->global_op_register_.insert({nncir::NodeType::ATENFILL, executorAtenFill});
    this->global_op_register_.insert({nncir::NodeType::ATENFLOORDIVIDE, executorAtenFloorDivide});
    this->global_op_register_.insert({nncir::NodeType::ATENFORMAT, executorAtenFormat});
    this->global_op_register_.insert({nncir::NodeType::ATENGETITEM, executorAtenGetItem});
    this->global_op_register_.insert({nncir::NodeType::ATENGATHER, executorAtenGather});
    this->global_op_register_.insert({nncir::NodeType::ATENGE, executorAtenGe});
    this->global_op_register_.insert({nncir::NodeType::ATENGT, executorAtenGt});
    this->global_op_register_.insert({nncir::NodeType::ATENINDEX, executorAtenIndex});
    this->global_op_register_.insert({nncir::NodeType::ATENINDEXPUT, executorAtenIndexPut});
    this->global_op_register_.insert({nncir::NodeType::ATENINDEXSELECT, executorAtenIndexSelect});
    this->global_op_register_.insert({nncir::NodeType::ATENINT, executorAtenInt});
    this->global_op_register_.insert({nncir::NodeType::ATENITEM, executorAtenItem});
    this->global_op_register_.insert({nncir::NodeType::ATENIS, executorAtenIs});
    this->global_op_register_.insert({nncir::NodeType::ATENLEAKYRELU, executorAtenLeakyRelu});
    this->global_op_register_.insert({nncir::NodeType::ATENLEN, executorAtenLen});
    this->global_op_register_.insert({nncir::NodeType::ATENLINEAR, executorAtenLinear});
    this->global_op_register_.insert({nncir::NodeType::ATENLIST, executorAtenList});
    this->global_op_register_.insert({nncir::NodeType::ATENLOG, executorAtenLog});
    this->global_op_register_.insert({nncir::NodeType::ATENLOGSOFTMAX, executorAtenLogSoftmax});
    this->global_op_register_.insert({nncir::NodeType::ATENLSTM, executorAtenLSTM});
    this->global_op_register_.insert({nncir::NodeType::ATENLT, executorAtenLt});
    this->global_op_register_.insert({nncir::NodeType::ATENMASKEDFILL, executorAtenMaskedFill});
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
    // output_tensors.clear();
    std::vector<torch::Tensor> out_tensors;
    std::vector<std::vector<int64_t>> out_list;

    // checkout the dtype of outpus
    bool is_all_tensor = true;
    for (auto& id_ : this->output_blob_ids_) {
        auto iv = this->findBlob(id_).second;
        if (!iv.isTensor()) {
            is_all_tensor = false;
            break;
        }
    }

    if (is_all_tensor) {
        for (auto& id_ : this->output_blob_ids_) {
            auto blob = this->findBlob(id_);
            output_tensors.push_back(blob.second.toTensor());
        }
    } else {
        // For RNNT, there's only one output with Tuple dtype
        //  %774 : (Tensor, Tensor, int[][]) = prim::TupleConstruct(%x_padded3.1, %x_lens.1, %output.1)
        //  return (%774)
        if (this->output_blob_ids_.size() == 1) {
            auto blob = this->findBlob(output_blob_ids_.at(0));
            if (blob.second.isTuple()) {
                auto tuple_ = blob.second.toTuple();
                auto ivs = primTupleUnpack(tuple_);
                for (auto& iv_ : ivs) {
                    if (iv_.isTensor()) {
                        auto tensor = iv_.toTensor();
                        out_tensors.push_back(tensor);
                    } else if (iv_.isList()) {
                        // list[list]
                        auto lst = iv_.toList().vec();
                        for (auto item : iv_.toList().vec()) {
                            std::vector<int64_t> vals;
                            for (auto val : item.toList().vec()) {
                                vals.push_back(val.toInt());
                            }
                            out_list.push_back(vals);
                        }
                    }
                }
            }

            // print RNNT result
            // logits, logits_lens, output
            // _, _, transcript = self.greedy_decoder.forward(feature, feature_length)

            std::stringstream ss;
            for (auto& item : out_list.at(0)) {
                ss << item << " ";
            }
            DLOG(INFO) << "RNNT_output: " << ss.str();
        }
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
