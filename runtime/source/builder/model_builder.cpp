/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#include "builder/model_builder.h"

namespace nn_compiler
{
namespace runtime
{
RetVal ModelBuilder::preProcess(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
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

RetVal ModelBuilder::preloadModel(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "ModelBuilder::preloadModel() is called";

    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        auto type = layer->getType();
        if (type == nn_compiler::ir::LayerType::ATENLSTM1 || type == nn_compiler::ir::LayerType::ATENLSTM2 ||
            type == nn_compiler::ir::LayerType::ATENCONV2D || type == nn_compiler::ir::LayerType::ATENBATCHNORM2D ||
            type == nn_compiler::ir::LayerType::ATENLINEAR || type == nn_compiler::ir::LayerType::ATENLAYERNORM) {
            // For Ops' with weight/bias, preload weights/bias to data_container.
            if (type == nn_compiler::ir::LayerType::ATENLSTM1) {
                auto lstm1_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
                auto ids = this->loadWeightAndBias(lstm1_layer->getWeights(), lstm1_layer->getBiases());
                lstm1_layer->setWeightIds(ids.first);
                lstm1_layer->setBiasIds(ids.second);
                if (feasibleForLstmWeightProcess(layer)) {
                    buildLstmParameterVector(layer);
                    reArangeLstmWeights(layer);
                } else {
                    DLOG(FATAL) << "Failed to pre-process weights for: " << layer->getName();
                }
            } else if (type == nn_compiler::ir::LayerType::ATENLSTM2) {
                auto lstm2_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer);
                auto ids = this->loadWeightAndBias(lstm2_layer->getWeights(), lstm2_layer->getBiases());
                lstm2_layer->setWeightIds(ids.first);
                lstm2_layer->setBiasIds(ids.second);
                if (feasibleForLstmWeightProcess(layer)) {
                    buildLstmParameterVector(layer);
                    reArangeLstmWeights(layer);
                } else {
                    DLOG(FATAL) << "Failed to pre-process weights for: " << layer->getName();
                }
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
            } else if (type == nn_compiler::ir::LayerType::ATENLAYERNORM) {
                auto layer_norm_layer = std::dynamic_pointer_cast<ir::AtenLayerNormLayer>(layer);
                auto ids = this->loadWeightAndBias(layer_norm_layer->getWeights(), layer_norm_layer->getBiases());
                layer_norm_layer->setWeightIds(ids.first);
                layer_norm_layer->setBiasIds(ids.second);
            }
        }
    }

    return RetVal::SUCCESS;
}

std::pair<std::vector<int64_t>, std::vector<int64_t> > ModelBuilder::loadWeightAndBias(
    std::vector<at::Tensor> weight_data, std::vector<at::Tensor> bias_data)
{
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

bool ModelBuilder::feasibleForLstmWeightProcess(std::shared_ptr<nn_compiler::ir::NNLayer>& layer)
{
    if (layer->getType() == nn_compiler::ir::LayerType::ATENLSTM1) {
        auto lstm1_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
        int has_biases = lstm1_layer->getHasBiases();
        int64_t num_layers = lstm1_layer->getNumLayers();
        double dropout = lstm1_layer->getDropout();
        int train = lstm1_layer->getTrain();
        int bidirectional = lstm1_layer->getBidirectional();
        int batch_first = lstm1_layer->getBatchFirst();
        if (nn_compiler::ir::isDefaultValue(has_biases) || nn_compiler::ir::isDefaultValue(num_layers) ||
            nn_compiler::ir::isDefaultValue(dropout) || nn_compiler::ir::isDefaultValue(train) ||
            nn_compiler::ir::isDefaultValue(bidirectional) || nn_compiler::ir::isDefaultValue(batch_first)) {
            return false;
        } else {
            return true;
        }
    } else if (layer->getType() == nn_compiler::ir::LayerType::ATENLSTM2) {
        auto lstm2_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer);
        int has_biases = lstm2_layer->getHasBiases();
        int64_t num_layers = lstm2_layer->getNumLayers();
        double dropout = lstm2_layer->getDropout();
        int train = lstm2_layer->getTrain();
        int bidirectional = lstm2_layer->getBidirectional();
        if (nn_compiler::ir::isDefaultValue(has_biases) || nn_compiler::ir::isDefaultValue(num_layers) ||
            nn_compiler::ir::isDefaultValue(dropout) || nn_compiler::ir::isDefaultValue(train) ||
            nn_compiler::ir::isDefaultValue(bidirectional)) {
            return false;
        } else {
            return true;
        }
    } else {
        DLOG(FATAL) << "Cannot pre-process weight for: " << ir::convertLayerTypeToString(layer->getType());
    }
}

RetVal ModelBuilder::buildLstmParameterVector(std::shared_ptr<nn_compiler::ir::NNLayer>& layer)
{
    int has_biases = 0;
    int64_t num_layers = 0;
    int bidirectional = 0;
    std::vector<int64_t> weight_ids, bias_ids;
    std::vector<at::Tensor> param_vector;
    auto layer_type = layer->getType();

    if (layer_type == nn_compiler::ir::LayerType::ATENLSTM1) {
        auto lstm1_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
        has_biases = lstm1_layer->getHasBiases();
        num_layers = lstm1_layer->getNumLayers();
        bidirectional = lstm1_layer->getBidirectional();
        weight_ids = lstm1_layer->getWeightIds();
        bias_ids = lstm1_layer->getBiasIds();
    } else if (layer_type == nn_compiler::ir::LayerType::ATENLSTM2) {
        auto lstm2_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer);
        has_biases = lstm2_layer->getHasBiases();
        num_layers = lstm2_layer->getNumLayers();
        bidirectional = lstm2_layer->getBidirectional();
        weight_ids = lstm2_layer->getWeightIds();
        bias_ids = lstm2_layer->getBiasIds();
    } else {
        DLOG(FATAL) << "Cannot build parameter vector for: " << ir::convertLayerTypeToString(layer_type);
    }

    // at::TensorList params
    // param layerout                --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?) * layers
    // param layerout (bidirctional) --> (fw_w_ih, fw_w_hh, fw_b_ih?, fw_b_hh?, bw_w_ih, bw_w_hh, bw_b_ih?, bw_b_hh?) *
    // layers
    assert((bidirectional == 0 || bidirectional == 1));
    for (int i = 0; i < num_layers * (bidirectional + 1); i++) {
        // w_ih
        auto w_ih_iv = (preloaded_data_container_.find(weight_ids[i * 2])->second).second;
        if (w_ih_iv.isTensor()) {
            param_vector.push_back(w_ih_iv.toTensor().cuda());
        }
        // w_hh
        auto w_hh_iv = (preloaded_data_container_.find(weight_ids[i * 2 + 1])->second).second;
        if (w_hh_iv.isTensor()) {
            param_vector.push_back(w_hh_iv.toTensor().cuda());
        }
        if (has_biases) {
            // b_ih
            auto b_ih_iv = (preloaded_data_container_.find(bias_ids[i * 2])->second).second;
            if (b_ih_iv.isTensor()) {
                param_vector.push_back(b_ih_iv.toTensor().cuda());
            }
            // b_hh
            auto b_hh_iv = (preloaded_data_container_.find(bias_ids[i * 2 + 1])->second).second;
            if (b_hh_iv.isTensor()) {
                param_vector.push_back(b_hh_iv.toTensor().cuda());
            }
        }
    }

    if (layer_type == nn_compiler::ir::LayerType::ATENLSTM1) {
        auto lstm1_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
        lstm1_layer->setParamVector(param_vector);
    } else {
        auto lstm2_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer);
        lstm2_layer->setParamVector(param_vector);
    }

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::reArangeLstmWeights(std::shared_ptr<nn_compiler::ir::NNLayer>& layer)
{
    int has_biases = 0;
    int64_t num_layers = 0;
    double dropout = 0.0;
    int train = 0;
    int bidirectional = 0;
    int bidirectional_int = 0;
    int batch_first = 0;
    std::vector<at::Tensor> param_vector;
    auto layer_type = layer->getType();

    if (layer_type == nn_compiler::ir::LayerType::ATENLSTM1) {
        auto lstm1_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
        has_biases = lstm1_layer->getHasBiases();
        num_layers = lstm1_layer->getNumLayers();
        dropout = lstm1_layer->getDropout();
        train = lstm1_layer->getTrain();
        bidirectional = lstm1_layer->getBidirectional();
        bidirectional_int = static_cast<bool>(bidirectional) ? 2 : 1;
        batch_first = lstm1_layer->getBatchFirst();
        param_vector = lstm1_layer->getParamVector();
    } else if (layer_type == nn_compiler::ir::LayerType::ATENLSTM2) {
        auto lstm2_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer);
        has_biases = lstm2_layer->getHasBiases();
        num_layers = lstm2_layer->getNumLayers();
        dropout = lstm2_layer->getDropout();
        train = lstm2_layer->getTrain();
        bidirectional = lstm2_layer->getBidirectional();
        bidirectional_int = static_cast<bool>(bidirectional) ? 2 : 1;
        param_vector = lstm2_layer->getParamVector();
    } else {
        DLOG(FATAL) << "Cannot re-arange parameter vector for: " << ir::convertLayerTypeToString(layer_type);
    }

    size_t weight_size = 0;

    size_t offset = 0;
    size_t param_size = 0;
    int data_size = 2;  // half

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
    auto weight_buffer = at::empty(weight_size / data_size, options);

    int param_num =
        static_cast<bool>(has_biases) ? 4 * num_layers * bidirectional_int : 2 * num_layers * bidirectional_int;

    int num_offset = static_cast<bool>(has_biases) ? 4 * bidirectional_int : 2 * bidirectional_int;
    for (int num = 0; num < param_num; num += num_offset) {
        param_size = 1;
        auto in_wei_vec = param_vector[num].sizes().vec();
        for (int i = 0; i < in_wei_vec.size(); ++i) {
            param_size *= in_wei_vec[i];
        }

        at::Tensor param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset,
                                         {static_cast<long>(param_size)}, weight_buffer.options());
        auto sliced_tensor = param_vector[num].chunk(4, 0);
        auto permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
        param.copy_(permuted_wei.view_as(param));
        offset += param_size;

        if (bidirectional_int == 2) {
            param_size = 1;
            in_wei_vec = param_vector[num + (num_offset / bidirectional_int)].sizes().vec();
            for (int i = 0; i < in_wei_vec.size(); ++i) {
                param_size *= in_wei_vec[i];
            }
            param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset, {static_cast<long>(param_size)},
                                  weight_buffer.options());
            sliced_tensor = param_vector[num + (num_offset / bidirectional_int)].chunk(4, 0);
            permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
            param.copy_(permuted_wei.view_as(param));
            offset += param_size;
        }

        param_size = 1;
        in_wei_vec = param_vector[num + 1].sizes().vec();
        for (int i = 0; i < in_wei_vec.size(); ++i) {
            param_size *= in_wei_vec[i];
        }
        param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset, {static_cast<long>(param_size)},
                              weight_buffer.options());
        sliced_tensor = param_vector[num + 1].chunk(4, 0);
        permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
        param.copy_(permuted_wei.view_as(param));
        offset += param_size;

        if (bidirectional_int == 2) {
            param_size = 1;
            in_wei_vec = param_vector[num + (num_offset / bidirectional_int) + 1].sizes().vec();
            for (int i = 0; i < in_wei_vec.size(); ++i) {
                param_size *= in_wei_vec[i];
            }
            param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset, {static_cast<long>(param_size)},
                                  weight_buffer.options());
            sliced_tensor = param_vector[num + (num_offset / bidirectional_int) + 1].chunk(4, 0);
            permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
            param.copy_(permuted_wei.view_as(param));
            offset += param_size;
        }
    }
    if (static_cast<bool>(has_biases)) {
        for (int num = 2; num < param_num; num += num_offset) {
            param_size = 1;
            auto in_wei_vec = param_vector[num].sizes().vec();
            for (int i = 0; i < in_wei_vec.size(); ++i) {
                param_size *= in_wei_vec[i];
            }
            at::Tensor param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset,
                                             {static_cast<long>(param_size)}, weight_buffer.options());
            auto sliced_tensor = param_vector[num].chunk(4, 0);
            auto permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
            param.copy_(permuted_wei.view_as(param));
            offset += param_size;

            if (bidirectional_int == 2) {
                param_size = 1;
                in_wei_vec = param_vector[num + (num_offset / bidirectional_int)].sizes().vec();
                for (int i = 0; i < in_wei_vec.size(); ++i) {
                    param_size *= in_wei_vec[i];
                }
                param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset, {static_cast<long>(param_size)},
                                      weight_buffer.options());
                sliced_tensor = param_vector[num + (num_offset / bidirectional_int)].chunk(4, 0);
                permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                param.copy_(permuted_wei.view_as(param));
                offset += param_size;
            }

            param_size = 1;
            in_wei_vec = param_vector[num + 1].sizes().vec();
            for (int i = 0; i < in_wei_vec.size(); ++i) {
                param_size *= in_wei_vec[i];
            }
            param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset, {static_cast<long>(param_size)},
                                  weight_buffer.options());
            sliced_tensor = param_vector[num + 1].chunk(4, 0);
            permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
            param.copy_(permuted_wei.view_as(param));
            offset += param_size;

            if (bidirectional_int == 2) {
                param_size = 1;
                in_wei_vec = param_vector[num + (num_offset / bidirectional_int) + 1].sizes().vec();
                for (int i = 0; i < in_wei_vec.size(); ++i) {
                    param_size *= in_wei_vec[i];
                }
                param = at::from_blob((_Float16*)(weight_buffer.data_ptr()) + offset, {static_cast<long>(param_size)},
                                      weight_buffer.options());
                sliced_tensor = param_vector[num + (num_offset / bidirectional_int) + 1].chunk(4, 0);
                permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
                param.copy_(permuted_wei.view_as(param));
                offset += param_size;
            }
        }
    }

    if (layer_type == nn_compiler::ir::LayerType::ATENLSTM1) {
        auto lstm1_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
        lstm1_layer->setArrangedWeight(weight_buffer);
    } else {
        auto lstm2_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer);
        lstm2_layer->setArrangedWeight(weight_buffer);
    }

    return RetVal::SUCCESS;
}

}  // namespace runtime
}  // namespace nn_compiler
