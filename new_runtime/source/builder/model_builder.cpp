#include "common/log.hpp"
#include "half.hpp"
#include "new_ir/include/layers/all_layers.h"
#include "new_ir/include/tensors/data_tensor.h"
#include "new_runtime/include/builder/model_builder.h"

namespace nn_compiler {
namespace runtime {

RetVal ModelBuilder::preProcess(std::unique_ptr<nn_compiler::ir::NNModel> &model) {
    Log::RT::I() << "ModelBuilder::preProcess() is called";

    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        auto out_stensor_ids = layer->getOutSTensorID();
        for (auto item : out_stensor_ids) {
            if (item > preload_start_id_) {
                preload_start_id_ = item;
            }
        }
    }
    preload_start_id_++;

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::preloadModel(std::unique_ptr<nn_compiler::ir::NNModel> &model) {
    Log::RT::I() << "ModelBuilder::preloadModel() is called";

    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        auto type = layer->getType();
        if (type == "aten::lstm1" || type == "aten::lstm2" || type == "aten::conv2d" ||
            type == "aten::batch_norm" || type == "aten::linear") {
            // For Ops' with weight/bias, preload weights/bias to data_container.
            std::vector<nn_compiler::ir::DTensor> weight_data;
            std::vector<nn_compiler::ir::DTensor> bias_data;

            if (type == "aten::lstm1") {
                auto lstm_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenLSTM1Layer>(layer);
                weight_data = lstm_layer->getWeights();
                bias_data = lstm_layer->getBiases();
            } else if (type == "aten::lstm2") {
                auto lstm_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenLSTM2Layer>(layer);
                weight_data = lstm_layer->getWeights();
                bias_data = lstm_layer->getBiases();
            } else if (type == "aten::conv2d") {
                auto conv2d_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenConv2dLayer>(layer);
                weight_data = conv2d_layer->getWeights();
                bias_data = conv2d_layer->getBiases();
            } else if (type == "aten::batch_norm") {
                auto bn2d_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenBatchNorm2dLayer>(layer);
                weight_data = bn2d_layer->getWeights();
                bias_data = bn2d_layer->getBiases();
            } else if (type == "aten::linear") {
                auto linear_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenLinearLayer>(layer);
                weight_data = linear_layer->getWeights();
                bias_data = linear_layer->getBiases();
            }

            for (auto idx = 0; idx < weight_data.size(); idx++) {
                this->loadWeightAndBias(weight_data[idx]);
            }
            for (auto idx = 0; idx < bias_data.size(); idx++) {
                this->loadWeightAndBias(bias_data[idx]);
            }
        }
    }

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::loadWeightAndBias(nn_compiler::ir::DTensor &data){
    auto this_id = preload_start_id_++;
    auto stensor = data.getTensorShape();
    auto bit_width = data.getBitWidth();

    at::ScalarType scalar_type;

    std::vector<int64_t> shape_arr;
    if (stensor.getBatch() > 0) shape_arr.push_back(stensor.getBatch());
    if (stensor.getChannel() > 0) shape_arr.push_back(stensor.getChannel());
    if (stensor.getHeight() > 0) shape_arr.push_back(stensor.getHeight());
    if (stensor.getWidth() > 0) shape_arr.push_back(stensor.getWidth());

    if (bit_width == 16) {
        auto value_vec = data.getData<half_float::half>();
        scalar_type = torch::kHalf;
        auto tensor_data = at::from_blob(value_vec->data(), shape_arr, scalar_type).cuda();
        torch::jit::IValue iv = torch::jit::IValue(tensor_data);
        this->preloaded_data_container_.insert({this_id, {DataType::TENSOR, iv}});
    } else if (bit_width == 32) {
        auto value_vec = data.getData<float>();
        scalar_type = torch::kFloat;
        auto tensor_data = at::from_blob(value_vec->data(), shape_arr, scalar_type).cuda();
        torch::jit::IValue iv = torch::jit::IValue(tensor_data);
        this->preloaded_data_container_.insert({this_id, {DataType::TENSOR, iv}});
    } else {
        Log::RT::E() << "Bit witdh Error!";
    }

    return RetVal::SUCCESS;
}

}  // namespace runtime
}  // namespace nn_compiler
