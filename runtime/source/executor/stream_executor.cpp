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
StreamExecutor::StreamExecutor(std::pair<std::shared_ptr<ir::NNNetwork>, blob_store_type> model,
                               std::string model_type)
{
    this->graph_ = model.first;
    this->global_blobs_ = model.second;
    this->undefined_data_ = std::make_pair(ir::DataType::UNDEFINED, intToIValue(0));
    this->setModelType(model_type);

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
        if (model_type_ == "GNMT" && layer->getType() == ir::LayerType::ATENCAT) {
            auto cat_layer = std::static_pointer_cast<ir::AtenCatLayer>(layer);
            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
            auto mem_blob_id = cat_layer->getMemLayerId();
            if (mem_blob_id != -1) {
                // memory cat_s
                auto cat_mem_s0 = at::zeros({1, 1, 2048}, options);
                this->global_blobs_.insert({mem_blob_id, {ir::DataType::TENSOR, tensorToIValue(cat_mem_s0)}});
            } else {
                // memory cat_f
                int cat_f = 22222;
                cat_layer->setMemLayerId(cat_f);
                auto cat_mem = at::empty({8, 1, 1024}, options);
                this->global_blobs_.insert({cat_f, {ir::DataType::TENSOR, tensorToIValue(cat_mem)}});
            }
        }

        if (layer->getType() == ir::LayerType::PRIMINPUT) {
            auto out_stensor_ids = layer->getOutSTensorID();
            input_blob_ids_.push_back(out_stensor_ids[0]);
        } else if (layer->getType() == ir::LayerType::PRIMOUTPUT) {
            auto in_stensor_ids = layer->getInSTensorID();
            output_blob_ids_.push_back(in_stensor_ids[0]);
        } else if (layer->getType() == ir::LayerType::PRIMCONSTANT) {
            // to speed up, runtime only load Constant data once, all constants are reused
            // in every inference forward
            op_executor::executePrimConstant(layer, *this);
        } else if (layer->getType() == ir::LayerType::PRIMVARIABLE) {
            auto variable_layer = std::static_pointer_cast<ir::PrimVariableLayer>(layer);
            if (variable_layer->getIsConstant()) {
                // to speed up, runtime only load variable data once, all constants inside are reused
                // in every inference forward
                op_executor::executePrimVariable(layer, *this);
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
    auto is_control_op = [](ir::LayerType type) {
        return (type == ir::LayerType::PRIMIF || type == ir::LayerType::PRIMENDIF ||
                type == ir::LayerType::PRIMLOOP || type == ir::LayerType::PRIMENDLOOP ||
                type == ir::LayerType::PRIMBLOCK);
    };

    // Execute Graph
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        auto layer = layers[cursor_];
        auto layer_type = layer->getType();
        DLOG(INFO) << "Layer id:" << layer->getID() << " name: " << layer->getName()
                   << " type: " << convertLayerTypeToString(layer_type);

        if (layer_type == ir::LayerType::PRIMCONSTANT) {
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
    auto is_control_op = [](ir::LayerType type) {
        return (type == ir::LayerType::PRIMIF || type == ir::LayerType::PRIMENDIF ||
                type == ir::LayerType::PRIMLOOP || type == ir::LayerType::PRIMENDLOOP ||
                type == ir::LayerType::PRIMBLOCK);
    };

    // Execute Graph
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        auto layer = layers[cursor_];
        auto layer_name = layer->getName();
        auto layer_type = layer->getType();
        DLOG(INFO) << "Layer id:" << layer->getID() << " name: " << layer->getName()
                   << " type: " << convertLayerTypeToString(layer_type);

        if (layer_type == ir::LayerType::PRIMCONSTANT) {
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
    if (it == this->global_blobs_.end()) {
        return undefined_data_;
    }
    return it->second;
}

bool StreamExecutor::checkValidBlobID(int64_t blob_id) { return (global_blobs_.find(blob_id) != global_blobs_.end()); }

OpExecutorFn StreamExecutor::findOpExecutor(ir::LayerType& type)
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
        updateBlob(id, ir::DataType::TENSOR, tensorToIValue(input_tensors.at(k++)));
    }
}

std::vector<torch::Tensor> StreamExecutor::iValueParser(torch::jit::IValue& iv)
{
    std::vector<torch::Tensor> out_tensor;
    std::vector<int64_t> out_list;

    if (iv.isTuple()) {
        auto tuple_ = iv.toTuple();
        auto ivs = op_executor::primTupleUnpack(tuple_);
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

const std::shared_ptr<ir::NNNetwork> StreamExecutor::getGraph() { return this->graph_; }

void StreamExecutor::registerOp()
{
    this->global_op_register_.insert({ir::LayerType::ATENADD, op_executor::executeAtenAdd});
    this->global_op_register_.insert({ir::LayerType::ATENADDMM, op_executor::executeAtenAddmm});
    this->global_op_register_.insert({ir::LayerType::ATENAND, op_executor::executeAtenAnd});
    this->global_op_register_.insert({ir::LayerType::ATENANY, op_executor::executeAtenAny});
    this->global_op_register_.insert({ir::LayerType::ATENAPPEND, op_executor::executeAtenAppend});
    this->global_op_register_.insert({ir::LayerType::ATENARANGE1, op_executor::executeAtenArange1});
    this->global_op_register_.insert({ir::LayerType::ATENARANGE2, op_executor::executeAtenArange2});
    this->global_op_register_.insert({ir::LayerType::ATENARANGE3, op_executor::executeAtenArange3});
    this->global_op_register_.insert({ir::LayerType::ATENASTENSOR, op_executor::executeAtenAsTensor});
    this->global_op_register_.insert({ir::LayerType::ATENBATCHNORM2D, op_executor::executeAtenBatchNorm2d});
    this->global_op_register_.insert({ir::LayerType::ATENBITWISENOT, op_executor::executeAtenBitwiseNot});
    this->global_op_register_.insert({ir::LayerType::ATENBMM, op_executor::executeAtenBmm});
    this->global_op_register_.insert({ir::LayerType::ATENBOOL, op_executor::executeAtenBool});
    this->global_op_register_.insert({ir::LayerType::ATENCAT, op_executor::executeAtenCat});
    this->global_op_register_.insert({ir::LayerType::ATENCEIL, op_executor::executeAtenCeil});
    this->global_op_register_.insert({ir::LayerType::ATENCHUNK, op_executor::executeAtenChunk});
    this->global_op_register_.insert({ir::LayerType::ATENCLAMP, op_executor::executeAtenClamp});
    this->global_op_register_.insert({ir::LayerType::ATENCLEAR, op_executor::executeAtenClear});
    this->global_op_register_.insert({ir::LayerType::ATENCONTIGUOUS, op_executor::executeAtenContiguous});
    this->global_op_register_.insert({ir::LayerType::ATENCONV2D, op_executor::executeAtenConv2d});
    this->global_op_register_.insert({ir::LayerType::ATENCOPY, op_executor::executeAtenCopy});
    this->global_op_register_.insert({ir::LayerType::ATENCPU, op_executor::executeAtenCpu});
    this->global_op_register_.insert({ir::LayerType::ATENCUDA, op_executor::executeAtenCuda});
    this->global_op_register_.insert({ir::LayerType::ATENDERIVEINDEX, op_executor::executeAtenDeriveIndex});
    this->global_op_register_.insert({ir::LayerType::ATENDIM, op_executor::executeAtenDim});
    this->global_op_register_.insert({ir::LayerType::ATENDIV, op_executor::executeAtenDiv});
    this->global_op_register_.insert({ir::LayerType::ATENDROPOUT, op_executor::executeAtenDropout});
    this->global_op_register_.insert({ir::LayerType::ATENEMBEDDING, op_executor::executeAtenEmbedding});
    this->global_op_register_.insert({ir::LayerType::ATENEQ, op_executor::executeAtenEq});
    this->global_op_register_.insert({ir::LayerType::ATENEQUAL, op_executor::executeAtenEqual});
    this->global_op_register_.insert({ir::LayerType::ATENEXPAND, op_executor::executeAtenExpand});
    this->global_op_register_.insert({ir::LayerType::ATENFILL, op_executor::executeAtenFill});
    this->global_op_register_.insert({ir::LayerType::ATENFLOORDIVIDE, op_executor::executeAtenFloorDivide});
    this->global_op_register_.insert({ir::LayerType::ATENFORMAT, op_executor::executeAtenFormat});
    this->global_op_register_.insert({ir::LayerType::ATENGETITEM, op_executor::executeAtenGetItem});
    this->global_op_register_.insert({ir::LayerType::ATENGATHER, op_executor::executeAtenGather});
    this->global_op_register_.insert({ir::LayerType::ATENGE, op_executor::executeAtenGe});
    this->global_op_register_.insert({ir::LayerType::ATENGT, op_executor::executeAtenGt});
    this->global_op_register_.insert({ir::LayerType::ATENINDEX, op_executor::executeAtenIndex});
    this->global_op_register_.insert({ir::LayerType::ATENINDEXPUT, op_executor::executeAtenIndexPut});
    this->global_op_register_.insert({ir::LayerType::ATENINDEXSELECT, op_executor::executeAtenIndexSelect});
    this->global_op_register_.insert({ir::LayerType::ATENINT, op_executor::executeAtenInt});
    this->global_op_register_.insert({ir::LayerType::ATENITEM, op_executor::executeAtenItem});
    this->global_op_register_.insert({ir::LayerType::ATENIS, op_executor::executeAtenIs});
    this->global_op_register_.insert({ir::LayerType::ATENLEAKYRELU, op_executor::executeAtenLeakyRelu});
    this->global_op_register_.insert({ir::LayerType::ATENLEN, op_executor::executeAtenLen});
    this->global_op_register_.insert({ir::LayerType::ATENLINEAR, op_executor::executeAtenLinear});
    this->global_op_register_.insert({ir::LayerType::ATENLIST, op_executor::executeAtenList});
    this->global_op_register_.insert({ir::LayerType::ATENLOG, op_executor::executeAtenLog});
    this->global_op_register_.insert({ir::LayerType::ATENLOGSOFTMAX, op_executor::executeAtenLogSoftmax});
    this->global_op_register_.insert({ir::LayerType::ATENLSTM1, op_executor::executeAtenLSTM1});
    this->global_op_register_.insert({ir::LayerType::ATENLSTM2, op_executor::executeAtenLSTM2});
    this->global_op_register_.insert({ir::LayerType::ATENLT, op_executor::executeAtenLt});
    this->global_op_register_.insert({ir::LayerType::ATENMASKEDFILL, op_executor::executeAtenMaskedFill});
    this->global_op_register_.insert({ir::LayerType::ATENMASKEDSELECT, op_executor::executeAtenMaskedSelect});
    this->global_op_register_.insert({ir::LayerType::ATENMATMUL, op_executor::executeAtenMatmul});
    this->global_op_register_.insert({ir::LayerType::ATENMAX, op_executor::executeAtenMax});
    this->global_op_register_.insert({ir::LayerType::ATENMAXPOOL2D, op_executor::executeAtenMaxPool2d});
    this->global_op_register_.insert({ir::LayerType::ATENMIN, op_executor::executeAtenMin});
    this->global_op_register_.insert({ir::LayerType::ATENMUL, op_executor::executeAtenMul});
    this->global_op_register_.insert({ir::LayerType::ATENNE, op_executor::executeAtenNe});
    this->global_op_register_.insert({ir::LayerType::ATENNEG, op_executor::executeAtenNeg});
    this->global_op_register_.insert({ir::LayerType::ATENNORM, op_executor::executeAtenNorm});
    this->global_op_register_.insert({ir::LayerType::ATENNOT, op_executor::executeAtenNot});
    this->global_op_register_.insert({ir::LayerType::ATENONES, op_executor::executeAtenOnes});
    this->global_op_register_.insert(
        {ir::LayerType::ATENPACKPADDEDSEQUENCE, op_executor::executeAtenPackPaddedSequence});
    this->global_op_register_.insert(
        {ir::LayerType::ATENPADPACKEDSEQUENCE, op_executor::executeAtenPadPackedSequence});
    this->global_op_register_.insert({ir::LayerType::ATENPOW, op_executor::executeAtenPow});
    this->global_op_register_.insert({ir::LayerType::ATENRELU, op_executor::executeAtenRelu});
    this->global_op_register_.insert({ir::LayerType::ATENRESHAPE, op_executor::executeAtenReshape});
    this->global_op_register_.insert({ir::LayerType::ATENSELECT, op_executor::executeAtenSelect});
    this->global_op_register_.insert({ir::LayerType::ATENSETITEM, op_executor::executeAtenSetItem});
    this->global_op_register_.insert({ir::LayerType::ATENSIZE, op_executor::executeAtenSize});
    this->global_op_register_.insert({ir::LayerType::ATENSLICE, op_executor::executeAtenSlice});
    this->global_op_register_.insert({ir::LayerType::ATENSOFTMAX, op_executor::executeAtenSoftmax});
    this->global_op_register_.insert({ir::LayerType::ATENSQUEEZE, op_executor::executeAtenSqueeze});
    this->global_op_register_.insert({ir::LayerType::ATENSUB, op_executor::executeAtenSub});
    this->global_op_register_.insert({ir::LayerType::ATENSUM, op_executor::executeAtenSum});
    this->global_op_register_.insert({ir::LayerType::ATENTANH, op_executor::executeAtenTanh});
    this->global_op_register_.insert({ir::LayerType::ATENTENSOR, op_executor::executeAtenTensor});
    this->global_op_register_.insert({ir::LayerType::ATENTRANSPOSE, op_executor::executeAtenTranspose});
    this->global_op_register_.insert({ir::LayerType::ATENTO1, op_executor::executeAtenTo1});
    this->global_op_register_.insert({ir::LayerType::ATENTO2, op_executor::executeAtenTo2});
    this->global_op_register_.insert({ir::LayerType::ATENTOPK, op_executor::executeAtenTopk});
    this->global_op_register_.insert({ir::LayerType::ATENUNSQUEEZE, op_executor::executeAtenUnsqueeze});
    this->global_op_register_.insert({ir::LayerType::ATENVIEW, op_executor::executeAtenView});
    this->global_op_register_.insert({ir::LayerType::ATENWARN, op_executor::executeAtenWarn});
    this->global_op_register_.insert({ir::LayerType::ATENZEROS, op_executor::executeAtenZeros});
    this->global_op_register_.insert({ir::LayerType::ATENZEROSLIKE, op_executor::executeAtenZerosLike});

    this->global_op_register_.insert({LayerType::PRIMBLOCK, op_executor::executePrimBlock});
    this->global_op_register_.insert({LayerType::PRIMCONSTANT, op_executor::executePrimConstant});
    this->global_op_register_.insert({LayerType::PRIMDATA, op_executor::executePrimData});
    this->global_op_register_.insert({LayerType::PRIMDEVICE, op_executor::executePrimDevice});
    this->global_op_register_.insert({LayerType::PRIMDTYPE, op_executor::executePrimDtype});
    this->global_op_register_.insert({LayerType::PRIMENDLOOP, op_executor::executePrimEndLoop});
    this->global_op_register_.insert({LayerType::PRIMIF, op_executor::executePrimIf});
    this->global_op_register_.insert({LayerType::PRIMGETATTR, op_executor::executePrimGetAttr});
    this->global_op_register_.insert({LayerType::PRIMENDIF, op_executor::executePrimEndIf});
    this->global_op_register_.insert({LayerType::PRIMLOOP, op_executor::executePrimLoop});
    this->global_op_register_.insert({LayerType::PRIMLOOPINDEX, op_executor::executePrimLoopIndex});
    this->global_op_register_.insert({LayerType::PRIMLISTCONSTRUCT, op_executor::executePrimListConstruct});
    this->global_op_register_.insert({LayerType::PRIMLISTUNPACK, op_executor::executePrimListUnpack});
    this->global_op_register_.insert({LayerType::PRIMRAISEEXCEPTION, op_executor::executePrimRaiseException});
    this->global_op_register_.insert({LayerType::PRIMSETATTR, op_executor::executePrimSetAttr});
    this->global_op_register_.insert({LayerType::PRIMTUPLECONSTRUCT, op_executor::executePrimTupleConstruct});
    this->global_op_register_.insert({LayerType::PRIMTUPLEINDEX, op_executor::executePrimTupleIndex});
    this->global_op_register_.insert({LayerType::PRIMTUPLEUNPACK, op_executor::executePrimTupleUnpack});
    this->global_op_register_.insert({LayerType::PRIMTYPE, op_executor::executePrimType});
    this->global_op_register_.insert({LayerType::PRIMUNCHECKEDCAST, op_executor::executePrimUncheckedCast});
    this->global_op_register_.insert({LayerType::PRIMUNINITIALIZED, op_executor::executePrimUninitialized});
    this->global_op_register_.insert({LayerType::PRIMVARIABLE, op_executor::executePrimVariable});
}

}  // namespace runtime
}  // namespace nn_compiler
