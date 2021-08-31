#include "executor/stream_executor.h"
#include <torch/script.h>
#include "c10/hip/HIPFunctions.h"
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
#include "tv_tools.h"

#include <sys/time.h>

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
/**
 * @brief Pre-load weight & bias, the tensor are saved into GPU
 *
 * @param blob blob to be saved
 */
void StreamExecutor::loadWeightAndBias(nncir::Blob* blob)
{
    nncir::Shape4D shape = blob->getShape();
    int64_t blob_id = blob->getId();

    auto data_blob = cast_if<nncir::DataBlob>(blob);
    if (data_blob == nullptr) {
        LOG(FATAL) << "The blob is not weight or bias blob!";
    }

    auto bit_width = blob->getBitWidth();
    at::ScalarType scalar_type;
    // torch::Tensor tensor_data;

    std::vector<int64_t> shape_arr;
    if (shape.n > 0) shape_arr.push_back(shape.n);
    if (shape.c > 0) shape_arr.push_back(shape.c);
    if (shape.h > 0) shape_arr.push_back(shape.h);
    if (shape.w > 0) shape_arr.push_back(shape.w);

    if (bit_width == 16) {
        auto value_vec = data_blob->getBuf<float16>();
        scalar_type = torch::kHalf;
        auto tensor_data = at::from_blob(value_vec.data(), shape_arr, scalar_type).cuda();
        // DLOG(INFO) << tensor_data;
        torch::jit::IValue iv = tensorToIValue(tensor_data);
        this->global_blobs_.insert({blob_id, {DataType::TENSOR, iv}});
    } else if (bit_width == 32) {
        auto value_vec = data_blob->getBuf<float>();
        scalar_type = torch::kFloat;
        auto tensor_data = at::from_blob(value_vec.data(), shape_arr, scalar_type).cuda();
        torch::jit::IValue iv = tensorToIValue(tensor_data);
        this->global_blobs_.insert({blob_id, {DataType::TENSOR, iv}});
    } else {
        LOG(FATAL) << "Bit witdh Error!";
    }
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
        } else if (op_node.getNodeType() == nncir::NodeType::ATENLSTM1 ||
                   op_node.getNodeType() == nncir::NodeType::ATENLSTM2 ||
                   op_node.getNodeType() == nncir::NodeType::ATENCONV2D ||
                   op_node.getNodeType() == nncir::NodeType::ATENBATCHNORM2D ||
                   op_node.getNodeType() == nncir::NodeType::ATENLINEAR) {
            // For Ops' with weight/bias, firstly save to global_blobs_ once
            std::vector<nncir::Blob*> weight_blobs;
            std::vector<nncir::Blob*> bias_blobs;

            if (op_node.getNodeType() == nncir::NodeType::ATENLSTM1) {
                auto lstm_node = cast<nncir::AtenLSTM1Node>(op_node);
                weight_blobs = lstm_node.getWeightBlob();
                bias_blobs = lstm_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENLSTM2) {
                auto lstm_node = cast<nncir::AtenLSTM2Node>(op_node);
                weight_blobs = lstm_node.getWeightBlob();
                bias_blobs = lstm_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENCONV2D) {
                auto conv2d_node = cast<nncir::AtenConv2dNode>(op_node);
                weight_blobs = conv2d_node.getWeightBlob();
                bias_blobs = conv2d_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENBATCHNORM2D) {
                auto bn2d_node = cast<nncir::AtenBatchNorm2dNode>(op_node);
                weight_blobs = bn2d_node.getWeightBlob();
                bias_blobs = bn2d_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENLINEAR) {
                auto linear_node = cast<nncir::AtenLinearNode>(op_node);
                weight_blobs = linear_node.getWeightBlob();
                bias_blobs = linear_node.getBiasBlob();
            }
            for (auto blob : weight_blobs) {
                this->loadWeightAndBias(blob);
            }
            for (auto blob : bias_blobs) {
                this->loadWeightAndBias(blob);
            }
        } else if (op_node.getNodeType() == nncir::NodeType::PRIMCONSTANT) {
            // to speed up, runtime only load Constant data once, all constants are reused
            // in every inference forward
            executePrimConstant(op_node, *this);
        }
    }

    DLOG(INFO) << "Num inputs of Graph:" << this->input_blob_ids_.size();
    DLOG(INFO) << "Num outputs of Graph:" << this->output_blob_ids_.size();

    if (this->input_blob_ids_.size() == 0 || this->output_blob_ids_.size() == 0) {
        LOG(FATAL) << "The Graph must have >= 1 inputs and outputs!";
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

        if (node_type == nncir::NodeType::PRIMCONSTANT) {
            // skip PrimConstant, constant are pre-loaded
            cursor_++;
            continue;
        } else {
            op_executor(*node, *this);
        }

        if (!is_control_op(node_type)) {
            cursor_++;
        }
    }
    // Read Output Tensors
    this->getOutputTensors(output_tensors);
    for (auto& out : output_tensors) {
        DLOG(INFO) << "Output Tensor:" << out.sizes();
    }

    return RetVal::SUCCESS;
}

RetVal StreamExecutor::inferenceModelwithProfiling(const std::shared_ptr<nncir::NNIR> graph,
                                                   const std::vector<torch::Tensor>& input_tensors,
                                                   std::vector<torch::Tensor>& output_tensors)
{
    bool enable_prof = false;
    timeval start, end;
    timeval h_start, h_end;
    float perf_getNode = 0.f;
    float perf_getNodeType = 0.f;
    float perf_findOpExecutor = 0.f;
    float perf_opExecutor = 0.f;
    std::unordered_map<std::string, float> perf_map;
    gettimeofday(&start, nullptr);

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

    gettimeofday(&end, nullptr);
    float perf = (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f;
    LOG(INFO) << "setInput perf is " << perf << " ms";

    // Execute Graph
    gettimeofday(&start, nullptr);
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        gettimeofday(&h_start, nullptr);
        nncir::Node* node = graph->getNode(cursor_);

        gettimeofday(&h_end, nullptr);
        perf_getNode += (h_end.tv_sec - h_start.tv_sec) * 1000.f + (h_end.tv_usec - h_start.tv_usec) / 1000.f;

        DLOG(INFO) << "Node id:" << node->getId() << " name:" << node->getName() << " type:" << node->getNodeType();

        gettimeofday(&h_start, nullptr);

        auto node_type = node->getNodeType();

        gettimeofday(&h_end, nullptr);
        perf_getNodeType += (h_end.tv_sec - h_start.tv_sec) * 1000.f + (h_end.tv_usec - h_start.tv_usec) / 1000.f;
        gettimeofday(&h_start, nullptr);

        auto op_executor = this->findOpExecutor(node_type);

        gettimeofday(&h_end, nullptr);
        perf_findOpExecutor += (h_end.tv_sec - h_start.tv_sec) * 1000.f + (h_end.tv_usec - h_start.tv_usec) / 1000.f;
        gettimeofday(&h_start, nullptr);

        if (node_type == nncir::NodeType::PRIMCONSTANT) {
            // skip PrimConstant, constant are pre-loaded
            cursor_++;
            continue;
        } else {
            op_executor(*node, *this);
        }

        if (!is_control_op(node_type)) {
            cursor_++;
        }

        at::hip::device_synchronize();
        gettimeofday(&h_end, nullptr);
        float tmp = (h_end.tv_sec - h_start.tv_sec) * 1000.f + (h_end.tv_usec - h_start.tv_usec) / 1000.f;
        perf_opExecutor += tmp;
        auto tnode = perf_map.find(node->getName());
        if (tnode != perf_map.end()) {
            tnode->second += tmp;
        } else {
            perf_map.insert(std::make_pair(node->getName(), tmp));
        }
    }

    gettimeofday(&end, nullptr);
    perf += (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f;
    LOG(INFO) << "execute graph perf is " << perf << " ms";
    LOG(INFO) << " ____perf_getNode is:        " << perf_getNode << " ms";
    LOG(INFO) << " ____perf_getNodeType is:    " << perf_getNodeType << " ms";
    LOG(INFO) << " ____perf_findOpExecutor is: " << perf_findOpExecutor << " ms";
    for (const auto& n : perf_map) {
        LOG(INFO) << " _____________________ " << n.first << "perf is : " << n.second << " ms";
    }
    LOG(INFO) << " perf_opExecutor is        " << perf_opExecutor << " ms";
    gettimeofday(&start, nullptr);

    // Read Output Tensors
    this->getOutputTensors(output_tensors);
    for (auto& out : output_tensors) {
        DLOG(INFO) << "Output Tensor:" << out.sizes();
    }

    gettimeofday(&end, nullptr);
    perf = (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f;
    LOG(INFO) << "setoutput perf is " << perf << " ms";

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

std::vector<torch::Tensor> StreamExecutor::iValueParser(torch::jit::IValue& iv)
{
    std::vector<torch::Tensor> out_tensor;
    std::vector<int64_t> out_list;
    if (iv.isTuple()) {
        auto tuple_ = iv.toTuple();
        auto ivs = primTupleUnpack(tuple_);
        for (auto iv_ : ivs) {
            if (iv_.isTensor()) {
                out_tensor.push_back(iv_.toTensor());
            } else if (iv_.isList()) {
                auto temp_out = iValueParser(iv_);
                for (auto& out : temp_out) {
                    out_tensor.push_back(out);
                }
            } else if (iv_.isInt()) {
                auto temp_out = iValueParser(iv_);
                for (auto& out : temp_out) {
                    out_tensor.push_back(out);
                }
            } else {
                LOG(FATAL) << "Dtype of output is unsupported !";
            }
        }
    } else if (iv.isList()) {
        auto list_ = iv.toListRef();
        out_list.clear();
        for (auto iv_ : list_) {
            if (iv_.isTensor()) {
                out_tensor.push_back(iv_.toTensor());
            } else if (iv_.isInt()) {
                out_list.push_back(iv_.toInt());
            } else if (iv_.isList()) {
                auto temp_out = iValueParser(iv_);
                for (auto& out : temp_out) {
                    out_tensor.push_back(out);
                }
            } else {
                LOG(FATAL) << "Dtype of output is unsupported !";
            }
        }
        if (out_list.size() != 0) {
            torch::Tensor out_ =
                torch::from_blob(out_list.data(), {1, static_cast<int64_t>(out_list.size())}, torch::kLong).clone();
            out_tensor.push_back(std::move(out_));
        }
    } else if (iv.isInt()) {
        out_list.push_back(iv.toInt());
        torch::Tensor out_ =
            torch::from_blob(out_list.data(), {static_cast<int64_t>(out_list.size())}, torch::kLong).clone();
        out_tensor.push_back(std::move(out_));
    } else if (iv.isTensor()) {
        out_tensor.push_back(iv.toTensor());
    } else {
        LOG(FATAL) << "Dtype of output is unsupported !";
    }
    return out_tensor;
}

void StreamExecutor::getOutputTensors(std::vector<torch::Tensor>& output_tensors)
{
    output_tensors.clear();
    std::vector<std::vector<int64_t>> out_list;

    for (auto& id_ : this->output_blob_ids_) {
        auto iv = this->findBlob(id_).second;
        auto temp_out = iValueParser(iv);
        for (auto& out : temp_out) {
            output_tensors.push_back(out);
        }
    }
    for (auto idx = 0; idx < output_tensors.size(); idx++) {
        DLOG(INFO) << "Output tensor" << idx << ": " << output_tensors[idx];
    }
}

const std::shared_ptr<nncir::NNIR> StreamExecutor::getGraph() { return this->ir_graph_; }

void StreamExecutor::registerOp()
{
    this->global_op_register_.insert({nncir::NodeType::ATENADD, executorAtenAdd});
    this->global_op_register_.insert({nncir::NodeType::ATENADDMM, executorAtenAddmm});
    this->global_op_register_.insert({nncir::NodeType::ATENAND, executorAtenAnd});
    this->global_op_register_.insert({nncir::NodeType::ATENANY, executorAtenAny});
    this->global_op_register_.insert({nncir::NodeType::ATENAPPEND, executorAtenAppend});
    this->global_op_register_.insert({nncir::NodeType::ATENARANGE1, executorAtenArange1});
    this->global_op_register_.insert({nncir::NodeType::ATENARANGE2, executorAtenArange2});
    this->global_op_register_.insert({nncir::NodeType::ATENARANGE3, executorAtenArange3});
    this->global_op_register_.insert({nncir::NodeType::ATENASTENSOR, executorAtenAsTensor});
    this->global_op_register_.insert({nncir::NodeType::ATENBATCHNORM2D, executorAtenBatchNorm2d});
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
    this->global_op_register_.insert({nncir::NodeType::ATENLSTM1, executorAtenLSTM1});
    this->global_op_register_.insert({nncir::NodeType::ATENLSTM2, executorAtenLSTM2});
    this->global_op_register_.insert({nncir::NodeType::ATENLT, executorAtenLt});
    this->global_op_register_.insert({nncir::NodeType::ATENMASKEDFILL, executorAtenMaskedFill});
    this->global_op_register_.insert({nncir::NodeType::ATENMASKEDSELECT, executorAtenMaskedSelect});
    this->global_op_register_.insert({nncir::NodeType::ATENMATMUL, executorAtenMatmul});
    this->global_op_register_.insert({nncir::NodeType::ATENMAX, executorAtenMax});
    this->global_op_register_.insert({nncir::NodeType::ATENMAXPOOL2D, executorAtenMaxPool2d});
    this->global_op_register_.insert({nncir::NodeType::ATENMIN, executorAtenMin});
    this->global_op_register_.insert({nncir::NodeType::ATENMUL, executorAtenMul});
    this->global_op_register_.insert({nncir::NodeType::ATENNE, executorAtenNe});
    this->global_op_register_.insert({nncir::NodeType::ATENNEG, executorAtenNeg});
    this->global_op_register_.insert({nncir::NodeType::ATENNORM, executorAtenNorm});
    this->global_op_register_.insert({nncir::NodeType::ATENNOT, executorAtenNot});
    this->global_op_register_.insert({nncir::NodeType::ATENONES, executorAtenOnes});
    this->global_op_register_.insert({nncir::NodeType::ATENPACKPADDEDSEQUENCE, executorAtenPackPaddedSequence});
    this->global_op_register_.insert({nncir::NodeType::ATENPADPACKEDSEQUENCE, executorAtenPadPackedSequence});
    this->global_op_register_.insert({nncir::NodeType::ATENPOW, executorAtenPow});
    this->global_op_register_.insert({nncir::NodeType::ATENRELU, executorAtenRelu});
    this->global_op_register_.insert({nncir::NodeType::ATENRESHAPE, executorAtenReshape});
    this->global_op_register_.insert({nncir::NodeType::ATENSELECT, executorAtenSelect});
    this->global_op_register_.insert({nncir::NodeType::ATENSETITEM, executorAtenSetItem});
    this->global_op_register_.insert({nncir::NodeType::ATENSIZE, executorAtenSize});
    this->global_op_register_.insert({nncir::NodeType::ATENSLICE, executorAtenSlice});
    this->global_op_register_.insert({nncir::NodeType::ATENSOFTMAX, executorAtenSoftmax});
    this->global_op_register_.insert({nncir::NodeType::ATENSQUEEZE, executorAtenSqueeze});
    this->global_op_register_.insert({nncir::NodeType::ATENSUB, executorAtenSub});
    this->global_op_register_.insert({nncir::NodeType::ATENSUM, executorAtenSum});
    this->global_op_register_.insert({nncir::NodeType::ATENTANH, executorAtenTanh});
    this->global_op_register_.insert({nncir::NodeType::ATENTENSOR, executorAtenTensor});
    this->global_op_register_.insert({nncir::NodeType::ATENTRANSPOSE, executorAtenTranspose});
    this->global_op_register_.insert({nncir::NodeType::ATENTO1, executorAtenTo1});
    this->global_op_register_.insert({nncir::NodeType::ATENTO2, executorAtenTo2});
    this->global_op_register_.insert({nncir::NodeType::ATENTOPK, executorAtenTopk});
    this->global_op_register_.insert({nncir::NodeType::ATENUNSQUEEZE, executorAtenUnsqueeze});
    this->global_op_register_.insert({nncir::NodeType::ATENVIEW, executorAtenView});
    this->global_op_register_.insert({nncir::NodeType::ATENWARN, executorAtenWarn});
    this->global_op_register_.insert({nncir::NodeType::ATENZEROS, executorAtenZeros});
    this->global_op_register_.insert({nncir::NodeType::ATENZEROSLIKE, executorAtenZerosLike});

    this->global_op_register_.insert({nncir::NodeType::PRIMBLOCK, executePrimBlock});
    this->global_op_register_.insert({nncir::NodeType::PRIMCONSTANT, executePrimConstant});
    this->global_op_register_.insert({nncir::NodeType::PRIMDATA, executePrimData});
    this->global_op_register_.insert({nncir::NodeType::PRIMDEVICE, executePrimDevice});
    this->global_op_register_.insert({nncir::NodeType::PRIMDTYPE, executePrimDtype});
    this->global_op_register_.insert({nncir::NodeType::PRIMENDLOOP, executePrimEndLoop});
    this->global_op_register_.insert({nncir::NodeType::PRIMIF, executePrimIf});
    this->global_op_register_.insert({nncir::NodeType::PRIMGETATTR, executePrimGetAttr});
    this->global_op_register_.insert({nncir::NodeType::PRIMENDIF, executePrimEndIf});
    this->global_op_register_.insert({nncir::NodeType::PRIMLOOP, executePrimLoop});
    this->global_op_register_.insert({nncir::NodeType::PRIMLOOPINDEX, executePrimLoopIndex});
    this->global_op_register_.insert({nncir::NodeType::PRIMLISTCONSTRUCT, executePrimListConstruct});
    this->global_op_register_.insert({nncir::NodeType::PRIMLISTUNPACK, executePrimListUnpack});
    this->global_op_register_.insert({nncir::NodeType::PRIMRAISEEXCEPTION, executePrimRaiseException});
    this->global_op_register_.insert({nncir::NodeType::PRIMSETATTR, executePrimSetAttr});
    this->global_op_register_.insert({nncir::NodeType::PRIMTUPLECONSTRUCT, executePrimTupleConstruct});
    this->global_op_register_.insert({nncir::NodeType::PRIMTUPLEINDEX, executePrimTupleIndex});
    this->global_op_register_.insert({nncir::NodeType::PRIMTUPLEUNPACK, executePrimTupleUnpack});
    this->global_op_register_.insert({nncir::NodeType::PRIMTYPE, executePrimType});
    this->global_op_register_.insert({nncir::NodeType::PRIMUNCHECKEDCAST, executePrimUncheckedCast});
    this->global_op_register_.insert({nncir::NodeType::PRIMUNINITIALIZED, executePrimUninitialized});
    this->global_op_register_.insert({nncir::NodeType::PRIMVARIABLE, executePrimVariable});
}

}  // namespace nnrt
