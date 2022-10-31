#include <sys/time.h>
#include <torch/script.h>

#include "c10/hip/HIPFunctions.h"
#include "executor/op_executor/aten_ops_executor.h"
#include "executor/op_executor/miopen_ops_executor.h"
#include "executor/op_executor/prim_ops_executor.h"
#include "executor/op_executor/stream_ops_executor.h"
#include "executor/stream_executor.h"
#include "executor/utils/profiler.h"

using namespace nn_compiler::runtime::utils;

namespace nn_compiler
{
namespace runtime
{
std::mutex StreamExecutor::stream_exec_mutex_;

StreamExecutor::StreamExecutor(std::pair<std::shared_ptr<ir::NNGraph>, blob_store_type> model, std::string model_type)
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

    // 1. set Aten Lib as default executor for Ops
    for (auto layer : graph_->getLayers()) {
        auto type = layer->getType();
        if (type == ir::LayerType::PRIMINPUT || type == ir::LayerType::PRIMOUTPUT) {
            continue;
        }
        this->op_executor_.insert({layer->getID(), findAtenExecutorByType(type)});
    }

    // 2. pre-load constants
    int custom_cat_mem_id = 0;
    for (auto layer : graph_->getLayers()) {
        if (model_type_ == "GNMT" && layer->getType() == ir::LayerType::ATENLSTM1) {
            auto lstm_layer = std::static_pointer_cast<ir::AtenLSTM1Layer>(layer);
            custom_cat_mem_id = lstm_layer->getCustomCatMemId();
        }
        if (model_type_ == "GNMT" && layer->getType() == ir::LayerType::ATENCAT) {
            auto cat_layer = std::static_pointer_cast<ir::AtenCatLayer>(layer);
            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
            auto mem_blob_id = cat_layer->getMemLayerId();
            if (mem_blob_id != -1) {
                auto cat_mem = at::zeros({1, 1, 2048}, options);
                this->global_blobs_.insert({mem_blob_id, {ir::DataType::TENSOR, tensorToIValue(cat_mem)}});
            } else {
                cat_layer->setMemLayerId(custom_cat_mem_id);
                auto cat_mem = at::empty({8, 1, 1024}, options);
                this->global_blobs_.insert({custom_cat_mem_id, {ir::DataType::TENSOR, tensorToIValue(cat_mem)}});
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

    // 3. set stream arguments
    int max_stream_num = 0;
    for (auto layer : graph_->getLayers()) {
        if (layer->getType() == ir::LayerType::MULTISTREAM) {
            auto multi_stream_layer = std::static_pointer_cast<ir::MultiStreamLayer>(layer);
            max_stream_num =
                multi_stream_layer->getLayerNum() > max_stream_num ? multi_stream_layer->getLayerNum() : max_stream_num;
        }
    }
    this->setStreamNum(max_stream_num);
    this->setStreams();

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

    // Execute Graph
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        auto layer = layers[cursor_];
        auto layer_type = layer->getType();
        auto layer_id = layer->getID();
        DLOG(INFO) << "Layer id:" << layer_id << " name: " << layer->getName()
                   << " type: " << convertLayerTypeToString(layer_type);

        if (layer_type == ir::LayerType::PRIMCONSTANT) {
            // skip PrimConstant, constant are pre-loaded
            cursor_++;
            continue;
        }

        auto op_executor = getOpExecutor(layer_id);
#ifdef SELECT_OPTIMAL_LIB
        // Once MIOpen is needed, its LstmExecutor must run firstly because it should maintain the tensor relations(by
        // pointer, which is not initialized in AtenExecutors) for the custom optimization for GNMT model.
        if (!has_select_optimal_lib_ && ir::isMIOpenSupportOp(layer_type)) {
            op_executor = findMIOpenExecutorByType(layer_type);
        }
#endif
        op_executor(layer, *this);

        // control_op will move cursor by itself
        if (!ir::isControlOp(layer_type)) {
            cursor_++;
        }

#ifdef SELECT_OPTIMAL_LIB
        if (!has_select_optimal_lib_ && ir::isMIOpenSupportOp(layer_type)) {
            layers_to_select_lib_.insert(layer_id);
        }
#endif
    }

#ifdef SELECT_OPTIMAL_LIB
    if (!has_select_optimal_lib_) {
        setOptimalLibForOpExecution();
        has_select_optimal_lib_ = true;
    }
#endif

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

    // Execute Graph
    for (cursor_ = cursor_begin; cursor_ < cursor_end;) {
        auto layer = layers[cursor_];
        auto layer_name = layer->getName();
        auto layer_type = layer->getType();
        auto layer_id = layer->getID();
        DLOG(INFO) << "Layer id:" << layer_id << " name: " << layer_name
                   << " type: " << convertLayerTypeToString(layer_type);

        if (layer_type == ir::LayerType::PRIMCONSTANT) {
            // skip PrimConstant, constant are pre-loaded
            cursor_++;
            continue;
        }

        auto op_executor = getOpExecutor(layer_id);
        PROFILE_SCOPE(layer_name);
        op_executor(layer, *this);
        at::hip::device_synchronize();

        // control_op will move cursor by itself
        if (!ir::isControlOp(layer_type)) {
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

OpExecutorFn StreamExecutor::findAtenExecutorByType(ir::LayerType& type)
{
    auto it = aten_executor_map_.find(type);
    if (it == aten_executor_map_.end()) {
        DLOG(FATAL) << "Op : " << convertLayerTypeToString(type) << " found no Aten support.";
    }
    assert(it != aten_executor_map_.end());
    return it->second;
}

OpExecutorFn StreamExecutor::findMIOpenExecutorByType(ir::LayerType& type)
{
    auto it = miopen_executor_map_.find(type);
    if (it == miopen_executor_map_.end()) {
        DLOG(FATAL) << "Op : " << convertLayerTypeToString(type) << " found no MIOpen support.";
    }
    assert(it != miopen_executor_map_.end());
    return it->second;
}

OpExecutorFn StreamExecutor::getOpExecutor(uint32_t& id)
{
    auto it = op_executor_.find(id);
    if (it == op_executor_.end()) {
        DLOG(FATAL) << "Runtime error, Id: " << id << " is out of size!";
    }
    assert(it != op_executor_.end());
    return it->second;
}

void StreamExecutor::setOptimalLibForOpExecution()
{
    DLOG(INFO) << "StreamExecutor::setOptimalLibForOpExecution() is called.";
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    for (auto layer_id : layers_to_select_lib_) {
        auto layer = graph_->getLayerByID(layer_id);
        auto layer_type = layer->getType();

        // for Ops with MIOpen support, compare which lib can help them run faster
        auto aten_op_executor = findAtenExecutorByType(layer_type);
        auto miopen_op_executor = findMIOpenExecutorByType(layer_type);

        aten_op_executor(layer, *this);  // warm up for aten executor, while miopen executor has been warmed up.

        start_time = std::chrono::high_resolution_clock::now();
        aten_op_executor(layer, *this);
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> aten_duration = end_time - start_time;  // us
        start_time = std::chrono::high_resolution_clock::now();
        miopen_op_executor(layer, *this);
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> miopen_duration = end_time - start_time;  // us

        if (miopen_duration.count() < aten_duration.count()) {
            op_executor_[layer_id] = miopen_op_executor;  // the Op can run faster with MIOpen lib
        }
    }
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
    std::vector<std::vector<int64_t>> out_list;

    for (auto id : output_blob_ids_) {
        auto iv = findBlob(id).second;
        auto temp_out = iValueParser(iv);
        for (auto out : temp_out) {
            std::lock_guard<std::mutex> lock(stream_exec_mutex_);
            output_tensors.push_back(out.to("cuda:0"));  // All output needs to be copied to cuda:0
        }
    }
    for (auto idx = 0; idx < output_tensors.size(); idx++) {
        DLOG(INFO) << "Output tensor" << idx << ": " << output_tensors[idx];
    }
}

const std::shared_ptr<ir::NNGraph> StreamExecutor::getGraph() { return this->graph_; }

void StreamExecutor::registerOp()
{
    this->aten_executor_map_.insert({ir::LayerType::ATENABS, op_executor::executeAtenAbs});
    this->aten_executor_map_.insert({ir::LayerType::ATENADD, op_executor::executeAtenAdd});
    this->aten_executor_map_.insert({ir::LayerType::ATENADDMM, op_executor::executeAtenAddmm});
    this->aten_executor_map_.insert({ir::LayerType::ATENAND, op_executor::executeAtenAnd});
    this->aten_executor_map_.insert({ir::LayerType::ATENANY, op_executor::executeAtenAny});
    this->aten_executor_map_.insert({ir::LayerType::ATENAPPEND, op_executor::executeAtenAppend});
    this->aten_executor_map_.insert({ir::LayerType::ATENARANGE1, op_executor::executeAtenArange1});
    this->aten_executor_map_.insert({ir::LayerType::ATENARANGE2, op_executor::executeAtenArange2});
    this->aten_executor_map_.insert({ir::LayerType::ATENARANGE3, op_executor::executeAtenArange3});
    this->aten_executor_map_.insert({ir::LayerType::ATENARGMAX, op_executor::executeAtenArgmax});
    this->aten_executor_map_.insert({ir::LayerType::ATENASTENSOR, op_executor::executeAtenAsTensor});
    this->aten_executor_map_.insert({ir::LayerType::ATENBATCHNORM2D, op_executor::executeAtenBatchNorm2d});
    this->aten_executor_map_.insert({ir::LayerType::ATENBITWISENOT, op_executor::executeAtenBitwiseNot});
    this->aten_executor_map_.insert({ir::LayerType::ATENBMM, op_executor::executeAtenBmm});
    this->aten_executor_map_.insert({ir::LayerType::ATENBOOL, op_executor::executeAtenBool});
    this->aten_executor_map_.insert({ir::LayerType::ATENCAT, op_executor::executeAtenCat});
    this->aten_executor_map_.insert({ir::LayerType::ATENCEIL, op_executor::executeAtenCeil});
    this->aten_executor_map_.insert({ir::LayerType::ATENCHUNK, op_executor::executeAtenChunk});
    this->aten_executor_map_.insert({ir::LayerType::ATENCLAMP, op_executor::executeAtenClamp});
    this->aten_executor_map_.insert({ir::LayerType::ATENCLEAR, op_executor::executeAtenClear});
    this->aten_executor_map_.insert({ir::LayerType::ATENCLONE, op_executor::executeAtenClone});
    this->aten_executor_map_.insert({ir::LayerType::ATENCONTIGUOUS, op_executor::executeAtenContiguous});
    this->aten_executor_map_.insert({ir::LayerType::ATENCONV2D, op_executor::executeAtenConv2d});
    this->aten_executor_map_.insert({ir::LayerType::ATENCOPY, op_executor::executeAtenCopy});
    this->aten_executor_map_.insert({ir::LayerType::ATENCPU, op_executor::executeAtenCpu});
    this->aten_executor_map_.insert({ir::LayerType::ATENCUDA, op_executor::executeAtenCuda});
    this->aten_executor_map_.insert({ir::LayerType::ATENCUMSUM, op_executor::executeAtenCumsum});
    this->aten_executor_map_.insert({ir::LayerType::ATENDERIVEINDEX, op_executor::executeAtenDeriveIndex});
    this->aten_executor_map_.insert({ir::LayerType::ATENDETACH, op_executor::executeAtenDetach});
    this->aten_executor_map_.insert({ir::LayerType::ATENDIM, op_executor::executeAtenDim});
    this->aten_executor_map_.insert({ir::LayerType::ATENDIV, op_executor::executeAtenDiv});
    this->aten_executor_map_.insert({ir::LayerType::ATENDROPOUT, op_executor::executeAtenDropout});
    this->aten_executor_map_.insert({ir::LayerType::ATENEINSUM, op_executor::executeAtenEinsum});
    this->aten_executor_map_.insert({ir::LayerType::ATENEMBEDDING, op_executor::executeAtenEmbedding});
    this->aten_executor_map_.insert({ir::LayerType::ATENEQ, op_executor::executeAtenEq});
    this->aten_executor_map_.insert({ir::LayerType::ATENEQUAL, op_executor::executeAtenEqual});
    this->aten_executor_map_.insert({ir::LayerType::ATENEXPAND, op_executor::executeAtenExpand});
    this->aten_executor_map_.insert({ir::LayerType::ATENFILL, op_executor::executeAtenFill});
    this->aten_executor_map_.insert({ir::LayerType::ATENFLOORDIVIDE, op_executor::executeAtenFloorDivide});
    this->aten_executor_map_.insert({ir::LayerType::ATENFORMAT, op_executor::executeAtenFormat});
    this->aten_executor_map_.insert({ir::LayerType::ATENFULLLIKE, op_executor::executeAtenFullLike});
    this->aten_executor_map_.insert({ir::LayerType::ATENGETITEM, op_executor::executeAtenGetItem});
    this->aten_executor_map_.insert({ir::LayerType::ATENGATHER, op_executor::executeAtenGather});
    this->aten_executor_map_.insert({ir::LayerType::ATENGE, op_executor::executeAtenGe});
    this->aten_executor_map_.insert({ir::LayerType::ATENGT, op_executor::executeAtenGt});
    this->aten_executor_map_.insert({ir::LayerType::ATENINDEX, op_executor::executeAtenIndex});
    this->aten_executor_map_.insert({ir::LayerType::ATENINDEXPUT, op_executor::executeAtenIndexPut});
    this->aten_executor_map_.insert({ir::LayerType::ATENINDEXSELECT, op_executor::executeAtenIndexSelect});
    this->aten_executor_map_.insert({ir::LayerType::ATENINT, op_executor::executeAtenInt});
    this->aten_executor_map_.insert({ir::LayerType::ATENINTIMPLICIT, op_executor::executeAtenIntImplicit});
    this->aten_executor_map_.insert({ir::LayerType::ATENITEM, op_executor::executeAtenItem});
    this->aten_executor_map_.insert({ir::LayerType::ATENIS, op_executor::executeAtenIs});
    this->aten_executor_map_.insert({ir::LayerType::ATENISINF, op_executor::executeAtenIsInf});
    this->aten_executor_map_.insert({ir::LayerType::ATENISNOT, op_executor::executeAtenIsNot});
    this->aten_executor_map_.insert({ir::LayerType::ATENLAYERNORM, op_executor::executeAtenLayerNorm});
    this->aten_executor_map_.insert({ir::LayerType::ATENLEAKYRELU, op_executor::executeAtenLeakyRelu});
    this->aten_executor_map_.insert({ir::LayerType::ATENLE, op_executor::executeAtenLe});
    this->aten_executor_map_.insert({ir::LayerType::ATENLEN, op_executor::executeAtenLen});
    this->aten_executor_map_.insert({ir::LayerType::ATENLINEAR, op_executor::executeAtenLinear});
    this->aten_executor_map_.insert({ir::LayerType::ATENLIST, op_executor::executeAtenList});
    this->aten_executor_map_.insert({ir::LayerType::ATENLOG, op_executor::executeAtenLog});
    this->aten_executor_map_.insert({ir::LayerType::ATENLOGSOFTMAX, op_executor::executeAtenLogSoftmax});
    this->aten_executor_map_.insert({ir::LayerType::ATENLSTM1, op_executor::executeAtenLSTM1});
    this->aten_executor_map_.insert({ir::LayerType::ATENLSTM2, op_executor::executeAtenLSTM2});
    this->aten_executor_map_.insert({ir::LayerType::ATENLT, op_executor::executeAtenLt});
    this->aten_executor_map_.insert({ir::LayerType::ATENMASKEDFILL, op_executor::executeAtenMaskedFill});
    this->aten_executor_map_.insert({ir::LayerType::ATENMASKEDSELECT, op_executor::executeAtenMaskedSelect});
    this->aten_executor_map_.insert({ir::LayerType::ATENMATMUL, op_executor::executeAtenMatmul});
    this->aten_executor_map_.insert({ir::LayerType::ATENMAX, op_executor::executeAtenMax});
    this->aten_executor_map_.insert({ir::LayerType::ATENMAXPOOL2D, op_executor::executeAtenMaxPool2d});
    this->aten_executor_map_.insert({ir::LayerType::ATENMEAN, op_executor::executeAtenMean});
    this->aten_executor_map_.insert({ir::LayerType::ATENMIN, op_executor::executeAtenMin});
    this->aten_executor_map_.insert({ir::LayerType::ATENMUL, op_executor::executeAtenMul});
    this->aten_executor_map_.insert({ir::LayerType::ATENNE, op_executor::executeAtenNe});
    this->aten_executor_map_.insert({ir::LayerType::ATENNEG, op_executor::executeAtenNeg});
    this->aten_executor_map_.insert({ir::LayerType::ATENNORM, op_executor::executeAtenNorm});
    this->aten_executor_map_.insert({ir::LayerType::ATENNOT, op_executor::executeAtenNot});
    this->aten_executor_map_.insert({ir::LayerType::ATENONEHOT, op_executor::executeAtenOneHot});
    this->aten_executor_map_.insert({ir::LayerType::ATENONES, op_executor::executeAtenOnes});
    this->aten_executor_map_.insert(
        {ir::LayerType::ATENPACKPADDEDSEQUENCE, op_executor::executeAtenPackPaddedSequence});
    this->aten_executor_map_.insert({ir::LayerType::ATENPADPACKEDSEQUENCE, op_executor::executeAtenPadPackedSequence});
    this->aten_executor_map_.insert({ir::LayerType::ATENPERMUTE, op_executor::executeAtenPermute});
    this->aten_executor_map_.insert({ir::LayerType::ATENPOW, op_executor::executeAtenPow});
    this->aten_executor_map_.insert({ir::LayerType::ATENRELU, op_executor::executeAtenRelu});
    this->aten_executor_map_.insert({ir::LayerType::ATENRESHAPE, op_executor::executeAtenReshape});
    this->aten_executor_map_.insert({ir::LayerType::ATENREMAINDER, op_executor::executeAtenRemainder});
    this->aten_executor_map_.insert({ir::LayerType::ATENREPEAT, op_executor::executeAtenRepeat});
    this->aten_executor_map_.insert({ir::LayerType::ATENRSQRT, op_executor::executeAtenRsqrt});
    this->aten_executor_map_.insert({ir::LayerType::ATENSELECT, op_executor::executeAtenSelect});
    this->aten_executor_map_.insert({ir::LayerType::ATENSETITEM, op_executor::executeAtenSetItem});
    this->aten_executor_map_.insert({ir::LayerType::ATENSIZE, op_executor::executeAtenSize});
    this->aten_executor_map_.insert({ir::LayerType::ATENSLICE, op_executor::executeAtenSlice});
    this->aten_executor_map_.insert({ir::LayerType::ATENSOFTMAX, op_executor::executeAtenSoftmax});
    this->aten_executor_map_.insert({ir::LayerType::ATENSQUEEZE, op_executor::executeAtenSqueeze});
    this->aten_executor_map_.insert({ir::LayerType::ATENSUB, op_executor::executeAtenSub});
    this->aten_executor_map_.insert({ir::LayerType::ATENSUM, op_executor::executeAtenSum});
    this->aten_executor_map_.insert({ir::LayerType::ATENTANH, op_executor::executeAtenTanh});
    this->aten_executor_map_.insert({ir::LayerType::ATENTENSOR, op_executor::executeAtenTensor});
    this->aten_executor_map_.insert({ir::LayerType::ATENTRANSPOSE, op_executor::executeAtenTranspose});
    this->aten_executor_map_.insert({ir::LayerType::ATENTO1, op_executor::executeAtenTo1});
    this->aten_executor_map_.insert({ir::LayerType::ATENTO2, op_executor::executeAtenTo2});
    this->aten_executor_map_.insert({ir::LayerType::ATENTO3, op_executor::executeAtenTo3});
    this->aten_executor_map_.insert({ir::LayerType::ATENTOPK, op_executor::executeAtenTopk});
    this->aten_executor_map_.insert({ir::LayerType::ATENTRIU, op_executor::executeAtenTriu});
    this->aten_executor_map_.insert({ir::LayerType::ATENTYPEAS, op_executor::executeAtenTypeAs});
    this->aten_executor_map_.insert({ir::LayerType::ATENUNSQUEEZE, op_executor::executeAtenUnsqueeze});
    this->aten_executor_map_.insert({ir::LayerType::ATENVIEW, op_executor::executeAtenView});
    this->aten_executor_map_.insert({ir::LayerType::ATENWARN, op_executor::executeAtenWarn});
    this->aten_executor_map_.insert({ir::LayerType::ATENWHERE, op_executor::executeAtenWhere});
    this->aten_executor_map_.insert({ir::LayerType::ATENZEROS, op_executor::executeAtenZeros});
    this->aten_executor_map_.insert({ir::LayerType::ATENZEROSLIKE, op_executor::executeAtenZerosLike});

    this->aten_executor_map_.insert({LayerType::PRIMBLOCK, op_executor::executePrimBlock});
    this->aten_executor_map_.insert({LayerType::PRIMCONSTANT, op_executor::executePrimConstant});
    this->aten_executor_map_.insert({LayerType::PRIMDATA, op_executor::executePrimData});
    this->aten_executor_map_.insert({LayerType::PRIMDEVICE, op_executor::executePrimDevice});
    this->aten_executor_map_.insert({LayerType::PRIMDTYPE, op_executor::executePrimDtype});
    this->aten_executor_map_.insert({LayerType::PRIMENDLOOP, op_executor::executePrimEndLoop});
    this->aten_executor_map_.insert({LayerType::PRIMIF, op_executor::executePrimIf});
    this->aten_executor_map_.insert({LayerType::PRIMGETATTR, op_executor::executePrimGetAttr});
    this->aten_executor_map_.insert({LayerType::PRIMENDIF, op_executor::executePrimEndIf});
    this->aten_executor_map_.insert({LayerType::PRIMLOOP, op_executor::executePrimLoop});
    this->aten_executor_map_.insert({LayerType::PRIMLOOPINDEX, op_executor::executePrimLoopIndex});
    this->aten_executor_map_.insert({LayerType::PRIMLISTCONSTRUCT, op_executor::executePrimListConstruct});
    this->aten_executor_map_.insert({LayerType::PRIMLISTUNPACK, op_executor::executePrimListUnpack});
    this->aten_executor_map_.insert({LayerType::PRIMRAISEEXCEPTION, op_executor::executePrimRaiseException});
    this->aten_executor_map_.insert({LayerType::PRIMSETATTR, op_executor::executePrimSetAttr});
    this->aten_executor_map_.insert({LayerType::PRIMTOLIST, op_executor::executePrimToList});
    this->aten_executor_map_.insert({LayerType::PRIMTUPLECONSTRUCT, op_executor::executePrimTupleConstruct});
    this->aten_executor_map_.insert({LayerType::PRIMTUPLEINDEX, op_executor::executePrimTupleIndex});
    this->aten_executor_map_.insert({LayerType::PRIMTUPLEUNPACK, op_executor::executePrimTupleUnpack});
    this->aten_executor_map_.insert({LayerType::PRIMTYPE, op_executor::executePrimType});
    this->aten_executor_map_.insert({LayerType::PRIMUNCHECKEDCAST, op_executor::executePrimUncheckedCast});
    this->aten_executor_map_.insert({LayerType::PRIMUNINITIALIZED, op_executor::executePrimUninitialized});
    this->aten_executor_map_.insert({LayerType::PRIMVARIABLE, op_executor::executePrimVariable});

    this->aten_executor_map_.insert({LayerType::MULTISTREAM, op_executor::executeMultiStream});

    this->miopen_executor_map_.insert({ir::LayerType::ATENLSTM1, op_executor::executeMIOpenLSTM1});
    this->miopen_executor_map_.insert({ir::LayerType::ATENLSTM2, op_executor::executeMIOpenLSTM2});
}

}  // namespace runtime
}  // namespace nn_compiler
