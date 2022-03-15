#include "half.hpp"
#include "ir/include/layers/all_layers.h"
#include "ir/include/tensors/data_tensor.h"
#include "runtime/include/builder/model_builder.h"

namespace nn_compiler
{
namespace runtime
{

RetVal ModelBuilder::preProcess(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    DLOG(INFO) << "ModelBuilder::preProcess() is called";

    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        auto out_stensor_ids = layer->getOutSTensorID();
        for (auto item : out_stensor_ids) {
            if (item > preload_id_) {
                preload_id_ = item;
            }
        }
    }
    preload_id_++;

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::preloadModel(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    DLOG(INFO) << "ModelBuilder::preloadModel() is called";

    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        auto type = layer->getType();
        if (type == nn_compiler::ir::LayerType::ATENLSTM1 || type == nn_compiler::ir::LayerType::ATENLSTM2 ||
            type == nn_compiler::ir::LayerType::ATENCONV2D || type == nn_compiler::ir::LayerType::ATENBATCHNORM2D ||
            type == nn_compiler::ir::LayerType::ATENLINEAR) {
            // For Ops' with weight/bias, preload weights/bias to data_container.
            if (type == nn_compiler::ir::LayerType::ATENLSTM1) {
                auto lstm1_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
                auto ids = this->loadWeightAndBias(lstm1_layer->getWeights(), lstm1_layer->getBiases());
                lstm1_layer->setWeightIds(ids.first);
                lstm1_layer->setBiasIds(ids.second);
            } else if (type == nn_compiler::ir::LayerType::ATENLSTM2) {
                auto lstm2_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer);
                auto ids = this->loadWeightAndBias(lstm2_layer->getWeights(), lstm2_layer->getBiases());
                lstm2_layer->setWeightIds(ids.first);
                lstm2_layer->setBiasIds(ids.second);
            } else if (type == nn_compiler::ir::LayerType::ATENCONV2D) {
                auto conv2d_layer = std::dynamic_pointer_cast<ir::AtenConv2dLayer>(layer);
                auto ids = this->loadWeightAndBias(conv2d_layer->getWeights(), conv2d_layer->getBiases());
                conv2d_layer->setWeightIds(ids.first);
                conv2d_layer->setBiasIds(ids.second);
            } else if (type == nn_compiler::ir::LayerType::ATENBATCHNORM2D) {
                auto bn2d_layer = std::dynamic_pointer_cast<ir::AtenBatchNorm2dLayer>(layer);
                auto ids = this->loadWeightAndBias(bn2d_layer->getWeights(), bn2d_layer->getBiases());
                bn2d_layer->setWeightIds(ids.first);
                bn2d_layer->setBiasIds(ids.second);
            } else if (type == nn_compiler::ir::LayerType::ATENLINEAR) {
                auto linear_layer = std::dynamic_pointer_cast<ir::AtenLinearLayer>(layer);
                auto ids = this->loadWeightAndBias(linear_layer->getWeights(), linear_layer->getBiases());
                linear_layer->setWeightIds(ids.first);
                linear_layer->setBiasIds(ids.second);
            }
        }
    }

    return RetVal::SUCCESS;
}

std::pair<std::vector<int64_t>, std::vector<int64_t> >
ModelBuilder::loadWeightAndBias(std::vector<at::Tensor> weight_data, std::vector<at::Tensor> bias_data) {
    std::vector<int64_t> weight_ids;
    std::vector<int64_t> bias_ids;

    for (auto data : weight_data) {
        auto this_id = preload_id_++;
        auto cuda_data = std::move(data.cuda());
        torch::jit::IValue iv = torch::jit::IValue(cuda_data);
        this->preloaded_data_container_.insert({this_id, {DataType::TENSOR, iv}});

        weight_ids.push_back(this_id);
    }
    for (auto data : bias_data) {
        auto this_id = preload_id_++;
        auto cuda_data = std::move(data.cuda());
        torch::jit::IValue iv = torch::jit::IValue(cuda_data);
        this->preloaded_data_container_.insert({this_id, {DataType::TENSOR, iv}});

        bias_ids.push_back(this_id);
    }

    return std::make_pair(weight_ids, bias_ids);
}

}  // namespace runtime
}  // namespace nn_compiler
