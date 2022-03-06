#include <sys/time.h>
#include <torch/script.h>

#include "c10/hip/HIPFunctions.h"
#include "common/log.hpp"
#include "new_ir/include/types.h"
#include "new_runtime/include/executor/aten_ops_executor.h"
#include "new_runtime/include/executor/prim_ops_executor.h"
#include "new_runtime/include/executor/profiler.h"
#include "new_runtime/include/executor/stream_executor.h"
#include "new_runtime/include/executor/utils.h"

namespace nn_compiler
{
namespace runtime
{
StreamExecutor::StreamExecutor(std::pair<std::shared_ptr<nn_compiler::ir::NNNetwork>, blob_store_type> model,
                               std::string model_type)
{
    this->ir_graph_ = model.first;
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

RetVal StreamExecutor::preProcess(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    input_blob_ids_.clear();
    output_blob_ids_.clear();

    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        if (model_type_ == "GNMT" && layer->getType() == nn_compiler::ir::LayerType::ATENCAT) {
            // TODO(SRCX): Implement this part with new IR.
            /***
            auto cat_node = cast_if<nncir::AtenCatNode>(op_node);
            auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
            auto mem_blob_id = cat_node->getMemBlobId();
            if (mem_blob_id != -1) {
                // memory cat_s
                auto cat_mem_s0 = at::zeros({1, 1, 2048}, options);
                this->global_blobs_.insert({mem_blob_id, {DataType::TENSOR, tensorToIValue(cat_mem_s0)}});
            } else {
                // memory cat_f
                int cat_f = 22222;
                cat_node->setMemBlobId(cat_f);
                auto cat_mem = at::empty({8, 1, 1024}, options);
                this->global_blobs_.insert({cat_f, {DataType::TENSOR, tensorToIValue(cat_mem)}});
            }
            ***/
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
            auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
            if (variable_layer->getIsConstant()) {
                // to speed up, runtime only load variable data once, all constants inside are reused
                // in every inference forward
                executePrimVariable(layer, *this);
            }
        }
    }

    Log::RT::D() << "Num inputs of Graph: " << input_blob_ids_.size();
    Log::RT::D() << "Num outputs of Graph: " << output_blob_ids_.size();

    if (input_blob_ids_.size() == 0 || output_blob_ids_.size() == 0) {
        Log::RT::E() << "The Graph must have >= 1 inputs and outputs!";
    }
}

RetVal StreamExecutor::inferenceModel(std::unique_ptr<nn_compiler::ir::NNModel>& model,
                                      const std::vector<torch::Tensor>& input_tensors,
                                      std::vector<torch::Tensor>& output_tensors)
{
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();

    // Set Input Tensors
    for (auto& in : input_tensors) {
        Log::RT::D() << "Input Tensor:" << in.sizes() << " data:" << in;
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
        Log::RT::D() << "Layer id:" << layer->getID() << " name: " << layer->getName()
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
        Log::RT::D() << "Output Tensor size: " << out.sizes();
    }

    return RetVal::SUCCESS;
}

RetVal StreamExecutor::inferenceModelwithProfiling(std::unique_ptr<nn_compiler::ir::NNModel>& model,
                                                   const std::vector<torch::Tensor>& input_tensors,
                                                   std::vector<torch::Tensor>& output_tensors)
{
    ProfileWriter::beginSession("start");

    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();

    // Set Input Tensors
    for (auto& in : input_tensors) {
        Log::RT::D() << "Input Tensor:" << in.sizes() << " data:" << in;
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
        Log::RT::D() << "Layer id:" << layer->getID() << " name: " << layer->getName()
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
        Log::RT::D() << "Output Tensor size: " << out.sizes();
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

OpExecutorFn StreamExecutor::findOpExecutor(nn_compiler::ir::LayerType& type)
{
    auto it = global_op_register_.find(type);
    if (it == global_op_register_.end()) {
        Log::RT::E() << "Runtime error, Unregistered Op found: " << convertLayerTypeToString(type);
        ;
    }
    assert(it != global_op_register_.end());
    return it->second;
}

void StreamExecutor::setInputTensors(const std::vector<torch::Tensor>& input_tensors)
{
    if (input_tensors.size() != input_blob_ids_.size()) {
        Log::RT::E() << "Num tensors must match the num inputs of Graph,"
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

    // TODO(SRCX): implementation

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
        Log::RT::I() << "Output tensor" << idx << ": " << output_tensors[idx];
    }
}

const std::shared_ptr<nn_compiler::ir::NNNetwork> StreamExecutor::getGraph() { return this->ir_graph_; }

void StreamExecutor::registerOp()
{
    this->global_op_register_.insert({LayerType::ATENRESHAPE, executorAtenReshape});

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
