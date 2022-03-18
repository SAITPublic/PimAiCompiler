#include <sys/time.h>
#include <torch/script.h>

#include "c10/hip/HIPFunctions.h"
#include "ir/include/types.h"
#include "runtime/include/executor/op_executor/aten_ops_executor.h"
#include "runtime/include/executor/op_executor/prim_ops_executor.h"
#include "runtime/include/executor/stream_executor.h"
#include "runtime/include/executor/utils/profiler.h"
#include "runtime/include/executor/utils/utils.h"

namespace nn_compiler
{
namespace runtime
{
StreamExecutor::StreamExecutor(std::pair<std::shared_ptr<nn_compiler::ir::NNNetwork>, blob_store_type> model,
                               std::string model_type)
{
    this->graph_ = model.first;
    this->global_blobs_ = model.second;

    model_type_ = model_type;

    miopenCreate(&handle_);
    miopenCreateTensorDescriptor(&input_tensor_);
    miopenCreateTensorDescriptor(&hidden_tensor_);
    miopenCreateTensorDescriptor(&weight_tensor_);
    miopenCreateTensorDescriptor(&output_tensor_);
    miopenCreateRNNDescriptor(&rnn_desc_);

    this->registerOp();
}

StreamExecutor::~StreamExecutor()
{
    miopenDestroyTensorDescriptor(output_tensor_);
    miopenDestroyTensorDescriptor(weight_tensor_);
    miopenDestroyTensorDescriptor(hidden_tensor_);
    miopenDestroyTensorDescriptor(input_tensor_);

    miopenDestroyRNNDescriptor(rnn_desc_);
    miopenDestroy(handle_);
}

RetVal StreamExecutor::preProcess()
{
    input_blob_ids_.clear();
    output_blob_ids_.clear();

    for (auto layer : graph_->getLayers()) {
        if (model_type_ == "GNMT" && layer->getType() == nn_compiler::ir::LayerType::ATENCAT) {
            auto cat_layer = std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(layer);
            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
            auto mem_blob_id = cat_layer->getMemLayerId();
            if (mem_blob_id != -1) {
                // memory cat_s
                auto cat_mem_s0 = at::zeros({1, 1, 2048}, options);
                this->global_blobs_.insert({mem_blob_id, {DataType::TENSOR, tensorToIValue(cat_mem_s0)}});
            } else {
                // memory cat_f
                int cat_f = 22222;
                cat_layer->setMemLayerId(cat_f);
                auto cat_mem = at::empty({8, 1, 1024}, options);
                this->global_blobs_.insert({cat_f, {DataType::TENSOR, tensorToIValue(cat_mem)}});
            }
        }

        if (layer->getType() == nn_compiler::ir::LayerType::PRIMINPUT) {
            auto out_stensor_ids = layer->getOutSTensorID();
            input_blob_ids_.push_back(out_stensor_ids[0]);
        } else if (layer->getType() == nn_compiler::ir::LayerType::PRIMOUTPUT) {
            auto in_stensor_ids = layer->getInSTensorID();
            output_blob_ids_.push_back(in_stensor_ids[0]);
        } else if (layer->getType() == nn_compiler::ir::LayerType::PRIMCONSTANT) {
            // to speed up, runtime only load Constant data once, all constants are reused
            // in every inference forward
            executePrimConstant(layer, *this);
        } else if (layer->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE) {
            auto variable_layer = std::static_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
            if (variable_layer->getIsConstant()) {
                // to speed up, runtime only load variable data once, all constants inside are reused
                // in every inference forward
                executePrimVariable(layer, *this);
            }
        }
    }

    DLOG(INFO) << "Num inputs of Graph: " << input_blob_ids_.size();
    DLOG(INFO) << "Num outputs of Graph: " << output_blob_ids_.size();

    if (input_blob_ids_.size() == 0 || output_blob_ids_.size() == 0) {
        DLOG(FATAL) << "The Graph must have >= 1 inputs and outputs!";
    }

    return RetVal::SUCCESS;
}

RetVal StreamExecutor::inferenceModel(const std::vector<torch::Tensor>& input_tensors,
                                      std::vector<torch::Tensor>& output_tensors)
{
    auto layers = graph_->getLayers();

    // Set Input Tensors
    for (auto& in : input_tensors) {
        DLOG(INFO) << "Input Tensor:" << in.sizes() << " data:" << in;
    }
    setInputTensors(input_tensors);

    // cursor skiped the InputNode and OutputNode
    // [cursor_begin, cursor_end)
    int cursor_begin = input_tensors.size();
    int cursor_end = layers.size() - output_blob_ids_.size();

    // control_op will move cursor by itself
    auto is_control_op = [](nn_compiler::ir::LayerType type) {
        return (type == nn_compiler::ir::LayerType::PRIMIF || type == nn_compiler::ir::LayerType::PRIMENDIF ||
                type == nn_compiler::ir::LayerType::PRIMLOOP || type == nn_compiler::ir::LayerType::PRIMENDLOOP ||
                type == nn_compiler::ir::LayerType::PRIMBLOCK);
    };

    // Execute Graph
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        auto layer = layers[cursor_];
        auto layer_type = layer->getType();
        DLOG(INFO) << "Layer id:" << layer->getID() << " name: " << layer->getName()
                     << " type: " << convertLayerTypeToString(layer_type);

        if (layer_type == nn_compiler::ir::LayerType::PRIMCONSTANT) {
            // skip PrimConstant, constant are pre-loaded
            cursor_++;
            continue;
        } else {
            auto op_executor = findOpExecutor(layer_type);
            op_executor(layer, *this);
        }

        if (!is_control_op(layer_type)) {
            cursor_++;
        }
    }

    // Read Output Tensors
    getOutputTensors(output_tensors);
    for (auto& out : output_tensors) {
        DLOG(INFO) << "Output Tensor size: " << out.sizes();
    }

    return RetVal::SUCCESS;
}

RetVal StreamExecutor::inferenceModelwithProfiling(const std::vector<torch::Tensor>& input_tensors,
                                                   std::vector<torch::Tensor>& output_tensors)
{
    ProfileWriter::beginSession("start");

    auto layers = graph_->getLayers();

    // Set Input Tensors
    for (auto& in : input_tensors) {
        DLOG(INFO) << "Input Tensor:" << in.sizes() << " data:" << in;
    }
    setInputTensors(input_tensors);

    // cursor skiped the InputNode and OutputNode
    // [cursor_begin, cursor_end)
    int cursor_begin = input_tensors.size();
    int cursor_end = layers.size() - output_blob_ids_.size();

    // control_op will move cursor by itself
    auto is_control_op = [](nn_compiler::ir::LayerType type) {
        return (type == nn_compiler::ir::LayerType::PRIMIF || type == nn_compiler::ir::LayerType::PRIMENDIF ||
                type == nn_compiler::ir::LayerType::PRIMLOOP || type == nn_compiler::ir::LayerType::PRIMENDLOOP ||
                type == nn_compiler::ir::LayerType::PRIMBLOCK);
    };

    // Execute Graph
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        auto layer = layers[cursor_];
        auto layer_name = layer->getName();
        auto layer_type = layer->getType();
        DLOG(INFO) << "Layer id:" << layer->getID() << " name: " << layer->getName()
                     << " type: " << convertLayerTypeToString(layer_type);

        if (layer_type == nn_compiler::ir::LayerType::PRIMCONSTANT) {
            // skip PrimConstant, constant are pre-loaded
            cursor_++;
            continue;
        } else {
            auto op_executor = findOpExecutor(layer_type);
            PROFILE_SCOPE(layer_name);
            op_executor(layer, *this);
            at::hip::device_synchronize();
        }

        if (!is_control_op(layer_type)) {
            cursor_++;
        }
    }

    ProfileWriter::endSession();
    ProfileWriter::get().generate_chrome_trace("trace.json");

    // Read Output Tensors
    getOutputTensors(output_tensors);
    for (auto& out : output_tensors) {
        DLOG(INFO) << "Output Tensor size: " << out.sizes();
    }

    return RetVal::SUCCESS;
}

void StreamExecutor::updateBlob(int64_t blob_id, DataType dtype, const torch::jit::IValue& iv)
{
    auto it = this->global_blobs_.find(blob_id);
    if (it == this->global_blobs_.end()) {
        this->global_blobs_.insert({blob_id, {dtype, iv}});
    } else {
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

bool StreamExecutor::checkValidBlobID(int64_t blob_id)
{
    return (global_blobs_.find(blob_id) != global_blobs_.end());
}

OpExecutorFn StreamExecutor::findOpExecutor(nn_compiler::ir::LayerType& type)
{
    auto it = global_op_register_.find(type);
    if (it == global_op_register_.end()) {
        DLOG(FATAL) << "Runtime error, Unregistered Op found: " << convertLayerTypeToString(type);
        ;
    }
    assert(it != global_op_register_.end());
    return it->second;
}

void StreamExecutor::setInputTensors(const std::vector<torch::Tensor>& input_tensors)
{
    if (input_tensors.size() != input_blob_ids_.size()) {
        DLOG(FATAL) << "Num tensors must match the num inputs of Graph,"
                     << "the Graph needs " << input_blob_ids_.size() << " inputs !";
    }
    // Set the input tensors to placeholder, assume all inputs & outputs are Tensor type
    int k = 0;
    for (auto id : input_blob_ids_) {
        updateBlob(id, DataType::TENSOR, tensorToIValue(input_tensors.at(k++)));
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
                DLOG(FATAL) << "Dtype of output is unsupported !";
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
                DLOG(FATAL) << "Dtype of output is unsupported !";
            }
        }
        if (out_list.size() != 0) {
            torch::Tensor out =
                torch::from_blob(out_list.data(), {1, static_cast<int64_t>(out_list.size())}, torch::kLong).clone();
            out_tensor.push_back(std::move(out));
        }
    } else if (iv.isInt()) {
        out_list.push_back(iv.toInt());
        torch::Tensor out =
            torch::from_blob(out_list.data(), {static_cast<int64_t>(out_list.size())}, torch::kLong).clone();
        out_tensor.push_back(std::move(out));
    } else if (iv.isTensor()) {
        out_tensor.push_back(iv.toTensor());
    } else {
        DLOG(FATAL) << "Dtype of output is unsupported !";
    }
    return out_tensor;
}

void StreamExecutor::getOutputTensors(std::vector<torch::Tensor>& output_tensors)
{
    output_tensors.clear();
    std::vector<std::vector<int64_t>> out_list;

    for (auto id : output_blob_ids_) {
        auto iv = findBlob(id).second;
        auto temp_out = iValueParser(iv);
        for (auto out : temp_out) {
            output_tensors.push_back(out);
        }
    }
    for (auto idx = 0; idx < output_tensors.size(); idx++) {
        DLOG(INFO) << "Output tensor" << idx << ": " << output_tensors[idx];
    }
}

const std::shared_ptr<nn_compiler::ir::NNNetwork> StreamExecutor::getGraph() { return this->graph_; }

void StreamExecutor::registerOp()
{
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENADD, executorAtenAdd});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENADDMM, executorAtenAddmm});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENAND, executorAtenAnd});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENANY, executorAtenAny});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENAPPEND, executorAtenAppend});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENARANGE1, executorAtenArange1});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENARANGE2, executorAtenArange2});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENARANGE3, executorAtenArange3});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENASTENSOR, executorAtenAsTensor});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENBATCHNORM2D, executorAtenBatchNorm2d});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENBITWISENOT, executorAtenBitwiseNot});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENBMM, executorAtenBmm});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENBOOL, executorAtenBool});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCAT, executorAtenCat});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCEIL, executorAtenCeil});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCHUNK, executorAtenChunk});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCLAMP, executorAtenClamp});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCLEAR, executorAtenClear});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCONTIGUOUS, executorAtenContiguous});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCONV2D, executorAtenConv2d});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCOPY, executorAtenCopy});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCPU, executorAtenCpu});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENCUDA, executorAtenCuda});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENDERIVEINDEX, executorAtenDeriveIndex});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENDIM, executorAtenDim});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENDIV, executorAtenDiv});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENDROPOUT, executorAtenDropout});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENEMBEDDING, executorAtenEmbedding});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENEQ, executorAtenEq});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENEQUAL, executorAtenEqual});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENEXPAND, executorAtenExpand});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENFILL, executorAtenFill});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENFLOORDIVIDE, executorAtenFloorDivide});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENFORMAT, executorAtenFormat});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENGETITEM, executorAtenGetItem});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENGATHER, executorAtenGather});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENGE, executorAtenGe});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENGT, executorAtenGt});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENINDEX, executorAtenIndex});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENINDEXPUT, executorAtenIndexPut});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENINDEXSELECT, executorAtenIndexSelect});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENINT, executorAtenInt});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENITEM, executorAtenItem});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENIS, executorAtenIs});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLEAKYRELU, executorAtenLeakyRelu});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLEN, executorAtenLen});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLINEAR, executorAtenLinear});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLIST, executorAtenList});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLOG, executorAtenLog});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLOGSOFTMAX, executorAtenLogSoftmax});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLSTM1, executorAtenLSTM1});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLSTM2, executorAtenLSTM2});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENLT, executorAtenLt});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENMASKEDFILL, executorAtenMaskedFill});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENMASKEDSELECT, executorAtenMaskedSelect});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENMATMUL, executorAtenMatmul});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENMAX, executorAtenMax});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENMAXPOOL2D, executorAtenMaxPool2d});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENMIN, executorAtenMin});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENMUL, executorAtenMul});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENNE, executorAtenNe});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENNEG, executorAtenNeg});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENNORM, executorAtenNorm});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENNOT, executorAtenNot});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENONES, executorAtenOnes});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENPACKPADDEDSEQUENCE, executorAtenPackPaddedSequence});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENPADPACKEDSEQUENCE, executorAtenPadPackedSequence});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENPOW, executorAtenPow});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENRELU, executorAtenRelu});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENRESHAPE, executorAtenReshape});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSELECT, executorAtenSelect});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSETITEM, executorAtenSetItem});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSIZE, executorAtenSize});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSLICE, executorAtenSlice});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSOFTMAX, executorAtenSoftmax});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSQUEEZE, executorAtenSqueeze});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSUB, executorAtenSub});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENSUM, executorAtenSum});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENTANH, executorAtenTanh});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENTENSOR, executorAtenTensor});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENTRANSPOSE, executorAtenTranspose});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENTO1, executorAtenTo1});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENTO2, executorAtenTo2});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENTOPK, executorAtenTopk});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENUNSQUEEZE, executorAtenUnsqueeze});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENVIEW, executorAtenView});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENWARN, executorAtenWarn});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENZEROS, executorAtenZeros});
    this->global_op_register_.insert({nn_compiler::ir::LayerType::ATENZEROSLIKE, executorAtenZerosLike});

    this->global_op_register_.insert({LayerType::PRIMBLOCK, executePrimBlock});
    this->global_op_register_.insert({LayerType::PRIMCONSTANT, executePrimConstant});
    this->global_op_register_.insert({LayerType::PRIMDATA, executePrimData});
    this->global_op_register_.insert({LayerType::PRIMDEVICE, executePrimDevice});
    this->global_op_register_.insert({LayerType::PRIMDTYPE, executePrimDtype});
    this->global_op_register_.insert({LayerType::PRIMENDLOOP, executePrimEndLoop});
    this->global_op_register_.insert({LayerType::PRIMIF, executePrimIf});
    this->global_op_register_.insert({LayerType::PRIMGETATTR, executePrimGetAttr});
    this->global_op_register_.insert({LayerType::PRIMENDIF, executePrimEndIf});
    this->global_op_register_.insert({LayerType::PRIMLOOP, executePrimLoop});
    this->global_op_register_.insert({LayerType::PRIMLOOPINDEX, executePrimLoopIndex});
    this->global_op_register_.insert({LayerType::PRIMLISTCONSTRUCT, executePrimListConstruct});
    this->global_op_register_.insert({LayerType::PRIMLISTUNPACK, executePrimListUnpack});
    this->global_op_register_.insert({LayerType::PRIMRAISEEXCEPTION, executePrimRaiseException});
    this->global_op_register_.insert({LayerType::PRIMSETATTR, executePrimSetAttr});
    this->global_op_register_.insert({LayerType::PRIMTUPLECONSTRUCT, executePrimTupleConstruct});
    this->global_op_register_.insert({LayerType::PRIMTUPLEINDEX, executePrimTupleIndex});
    this->global_op_register_.insert({LayerType::PRIMTUPLEUNPACK, executePrimTupleUnpack});
    this->global_op_register_.insert({LayerType::PRIMTYPE, executePrimType});
    this->global_op_register_.insert({LayerType::PRIMUNCHECKEDCAST, executePrimUncheckedCast});
    this->global_op_register_.insert({LayerType::PRIMUNINITIALIZED, executePrimUninitialized});
    this->global_op_register_.insert({LayerType::PRIMVARIABLE, executePrimVariable});
}

}  // namespace runtime
}  // namespace nn_compiler
