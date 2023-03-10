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

#include "executor/op_executor/miopen_ops_executor.h"

using namespace nn_compiler::runtime::utils;

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
void executeMIOpenLSTM1(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute MIOpen LSTM1 layer";

    auto lstm1_layer = std::static_pointer_cast<nn_compiler::ir::AtenLSTM1Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    // const at::Tensor &input
    auto input_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(input_iv.isTensor());
    auto input = input_iv.toTensor();

    // at::TensorList hx
    auto hx_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(hx_iv.isTensorList());
    auto hx_list_tensor = hx_iv.toTensorList();
    std::vector<at::Tensor> hx_list_tensor_vector;
    for (auto tensor : hx_list_tensor) {
        hx_list_tensor_vector.push_back(tensor);
    }
    at::TensorList hx(hx_list_tensor_vector);

    // Check and skip, will handle params after getting all arguments
    if (in_stensor_id.size() > in_id) {
        auto params_iv = stream_executor.findBlob(in_stensor_id[in_id]).second;
        if (params_iv.isTensorList()) {
            in_id++;
        }
    }

    // bool has_biases
    int has_biases = lstm1_layer->getHasBiases();
    if (nn_compiler::ir::isDefaultValue(has_biases)) {
        auto has_biases_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(has_biases_iv.isInt());
        has_biases = has_biases_iv.toInt();
    }

    // int64_t num_layers
    int64_t num_layers = lstm1_layer->getNumLayers();
    if (nn_compiler::ir::isDefaultValue(num_layers)) {
        auto num_layers_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(num_layers_iv.isInt());
        num_layers = num_layers_iv.toInt();
    }

    // double dropout
    double dropout = lstm1_layer->getDropout();
    if (nn_compiler::ir::isDefaultValue(dropout)) {
        auto dropout_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dropout_iv.isDouble());
        dropout = dropout_iv.toDouble();
    }

    // bool train
    int train = lstm1_layer->getTrain();
    if (nn_compiler::ir::isDefaultValue(train)) {
        auto train_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(train_iv.isInt());
        train = train_iv.toInt();
    }

    // bool bidirectional
    int bidirectional = lstm1_layer->getBidirectional();
    if (nn_compiler::ir::isDefaultValue(bidirectional)) {
        auto bidirectional_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(bidirectional_iv.isInt());
        bidirectional = bidirectional_iv.toInt();
    }

    // bool batch_first
    int batch_first = lstm1_layer->getBatchFirst();
    if (nn_compiler::ir::isDefaultValue(batch_first)) {
        auto batch_first_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(batch_first_iv.isInt());
        batch_first = batch_first_iv.toInt();
    }

    auto param_vector = lstm1_layer->getParamVector();

    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    assert(out_stensor_id.size() == 3);

    if (!input.is_contiguous()) input = input.contiguous();
    if (!static_cast<bool>(batch_first)) input = input.transpose(0, 1);
    void *in_dev, *hx_dev, *out_dev, *wei_dev, *cx_dev, *workspace_dev, *hy_dev, *cy_dev;

    auto lstm_tensor = stream_executor.getMiopenLstmTensors();
    lstm_tensor.input_tensors_->clear();
    lstm_tensor.output_tensors_->clear();

    int batch_size = 1;
    int in_dim = input.dim();
    int input_size = input.size(in_dim - 1);
    int seq_len = input.size(in_dim - 2);
    int bidirectional_int = static_cast<bool>(bidirectional) ? 2 : 1;

    int hx_dim = hx_list_tensor_vector[0].dim();
    int hidden_size = hx_list_tensor_vector[0].size(hx_dim - 1);

    std::vector<int> in_len({batch_size, input_size});
    std::vector<int> hid_len({bidirectional_int * (int)(num_layers), hidden_size});
    std::vector<int> out_len({bidirectional_int * hidden_size});

    int dims = 2;
    for (int i = 0; i < seq_len; i++) {
        std::array<int, 2> in_lens = {in_len[0], in_len.back()};
        miopenCreateTensorDescriptor(lstm_tensor.input_tensor_);
        miopenSetTensorDescriptor(*(lstm_tensor.input_tensor_), miopenHalf, dims, in_lens.data(), nullptr);
        lstm_tensor.input_tensors_->push_back(*(lstm_tensor.input_tensor_));

        std::array<int, 2> out_lens = {{in_len[0], out_len[0]}};
        miopenCreateTensorDescriptor(lstm_tensor.output_tensor_);
        miopenSetTensorDescriptor(*(lstm_tensor.output_tensor_), miopenHalf, dims, out_lens.data(), nullptr);
        lstm_tensor.output_tensors_->push_back(*(lstm_tensor.output_tensor_));
    }
    std::array<int, 3> hid_lens = {{hid_len[0], in_len[0], hid_len[1]}};
    miopenSetTensorDescriptor(*(lstm_tensor.hidden_tensor_), miopenHalf, 3, hid_lens.data(), nullptr);

    miopenRNNMode_t mode = miopenRNNMode_t::miopenLSTM;

    miopenRNNBiasMode_t biasMode = static_cast<bool>(has_biases) ? miopenRNNwithBias : miopenRNNNoBias;
    miopenRNNDirectionMode_t directionMode = bidirectional_int == 2 ? miopenRNNbidirection : miopenRNNunidirection;
    miopenRNNInputMode_t inMode = miopenRNNlinear;
    miopenRNNAlgo_t algo = miopenRNNdefault;

    miopenSetRNNDescriptor(stream_executor.getMIOpenRNNDesc(), hidden_size, num_layers, inMode, directionMode, mode,
                           biasMode, algo, miopenHalf);
    miopenGetRNNParamsDescriptor(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(),
                                 *(lstm_tensor.input_tensor_), *(lstm_tensor.weight_tensor_), miopenHalf);
    size_t workspace_size;
    miopenTensorDescriptor_t* inputs_ptr = (*(lstm_tensor.input_tensors_)).data();
    miopenGetRNNWorkspaceSize(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(), seq_len,
                              inputs_ptr, &workspace_size);
    auto workspace = at::empty(workspace_size, input.options().dtype(at::kByte));

    in_dev = input.data_ptr();
    auto arranged_weight = lstm1_layer->getArrangedWeight();
    wei_dev = arranged_weight.data_ptr();
    hx_dev = hx_list_tensor_vector[0].data_ptr();
    cx_dev = hx_list_tensor_vector[1].data_ptr();
    workspace_dev = workspace.data_ptr();

    miopenTensorDescriptor_t* outputs_ptr = (*(lstm_tensor.output_tensors_)).data();
    if (stream_executor.getModelType() == "GNMT" &&
        stream_executor.findBlob(out_stensor_id[0]).first != ir::DataType::UNDEFINED && seq_len == 1) {
        out_dev = stream_executor.findBlob(out_stensor_id[0]).second.toTensor().data_ptr();
        hy_dev = stream_executor.findBlob(out_stensor_id[1]).second.toTensor().data_ptr();
        cy_dev = stream_executor.findBlob(out_stensor_id[2]).second.toTensor().data_ptr();
        miopenRNNForwardInference(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(), seq_len,
                                  inputs_ptr, in_dev, *(lstm_tensor.hidden_tensor_), hx_dev,
                                  *(lstm_tensor.hidden_tensor_), cx_dev, *(lstm_tensor.weight_tensor_), wei_dev,
                                  outputs_ptr, out_dev, *(lstm_tensor.hidden_tensor_), hy_dev,
                                  *(lstm_tensor.hidden_tensor_), cy_dev, workspace_dev, workspace_size);
    } else {
        auto output = at::empty({in_len[0], seq_len, out_len[0]}, input.options());
        out_dev = output.data_ptr();
        at::Tensor hy, cy;

        size_t hc_y_size = hid_lens[0] * hid_lens[1] * hid_lens[2];
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        hy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);
        cy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);

        hy_dev = hy.data_ptr();
        cy_dev = cy.data_ptr();

        if (stream_executor.getModelType() == "GNMT" && lstm1_layer->getMatchCustomOpt()) {
            auto cat_mem = stream_executor.findBlob(lstm1_layer->getCustomCatMemId()).second.toTensor();
            auto custom_opt_number = lstm1_layer->getCustomOptNumber();

            if (custom_opt_number == 0) {
                hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()), {1, 1, 1024}, options);
                cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 1024, {1, 1, 1024}, options);
                hy_dev = hy.data_ptr();
                cy_dev = cy.data_ptr();
            } else if (custom_opt_number == 1) {
                hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 2048, {1, 1, 1024}, options);
                cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 3072, {1, 1, 1024}, options);
                hy_dev = hy.data_ptr();
                cy_dev = cy.data_ptr();
            } else if (custom_opt_number == 2) {
                hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 4096, {1, 1, 1024}, options);
                cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 5120, {1, 1, 1024}, options);
                hy_dev = hy.data_ptr();
                cy_dev = cy.data_ptr();
            } else if (custom_opt_number == 3) {
                hy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 6144, {1, 1, 1024}, options);
                cy = torch::from_blob((_Float16*)(cat_mem.data_ptr()) + 7168, {1, 1, 1024}, options);
                hy_dev = hy.data_ptr();
                cy_dev = cy.data_ptr();
            }
            if (custom_opt_number == 0 || custom_opt_number == 1) {
                auto out_layer = stream_executor.getGraph()->getLayerByPosition(
                    (layer->getNextLayerIDs())[custom_opt_number == 0 ? 3 : 0]);
                auto out_out_layer = stream_executor.getGraph()->getLayerByPosition((out_layer->getNextLayerIDs())[0]);
                int64_t cat_mem_id =
                    std::static_pointer_cast<nn_compiler::ir::AtenCatLayer>(out_out_layer)->getMemLayerId();
                auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
                auto cat_mem = stream_executor.findBlob(cat_mem_id).second.toTensor();
                output = torch::from_blob((_Float16*)(cat_mem.data_ptr()), {1, 1, 1024}, options);
            }
        }

        miopenRNNForwardInference(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(), seq_len,
                                  inputs_ptr, in_dev, *(lstm_tensor.hidden_tensor_), hx_dev,
                                  *(lstm_tensor.hidden_tensor_), cx_dev, *(lstm_tensor.weight_tensor_), wei_dev,
                                  outputs_ptr, out_dev, *(lstm_tensor.hidden_tensor_), hy_dev,
                                  *(lstm_tensor.hidden_tensor_), cy_dev, workspace_dev, workspace_size);

        if (!static_cast<bool>(batch_first)) output = output.transpose(0, 1);
        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(hy));
        stream_executor.updateBlob(out_stensor_id[2], DataType::TENSOR, tensorToIValue(cy));
    }
}

void executeMIOpenLSTM2(std::shared_ptr<nn_compiler::ir::NNLayer>& layer, StreamExecutor& stream_executor)
{
    DLOG(INFO) << "execute MIOpen LSTM2 layer";

    auto lstm2_layer = std::static_pointer_cast<nn_compiler::ir::AtenLSTM2Layer>(layer);

    auto in_stensor_id = layer->getInSTensorID();
    auto out_stensor_id = layer->getOutSTensorID();

    int in_id = 0;
    // const at::Tensor &input
    auto input_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(input_iv.isTensor());
    auto input = input_iv.toTensor();

    // const at::Tensor &batch_sizes,
    at::Tensor batch_sizes;
    auto batch_sizes_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(batch_sizes_iv.isTensor());
    batch_sizes = batch_sizes_iv.toTensor();

    // at::TensorList hx
    auto hx_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
    assert(hx_iv.isTensorList());
    auto hx_list_tensor = hx_iv.toTensorList();
    std::vector<at::Tensor> hx_list_tensor_vector;
    for (auto tensor : hx_list_tensor) {
        hx_list_tensor_vector.push_back(tensor);
    }
    at::TensorList hx(hx_list_tensor_vector);

    // at::TensorList params
    // Check and skip, will handle params after getting all arguments
    if (in_stensor_id.size() > 2) {
        auto params_iv = stream_executor.findBlob(in_stensor_id[in_id]).second;
        if (params_iv.isTensorList()) {
            in_id++;
        }
    }

    // bool has_biases
    int has_biases = lstm2_layer->getHasBiases();
    if (nn_compiler::ir::isDefaultValue(has_biases)) {
        auto has_biases_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(has_biases_iv.isInt());
        has_biases = has_biases_iv.toInt();
    }

    // int64_t num_layers
    int64_t num_layers = lstm2_layer->getNumLayers();
    if (nn_compiler::ir::isDefaultValue(num_layers)) {
        auto num_layers_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(num_layers_iv.isInt());
        num_layers = num_layers_iv.toInt();
    }

    // double dropout
    double dropout = lstm2_layer->getDropout();
    if (nn_compiler::ir::isDefaultValue(dropout)) {
        auto dropout_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(dropout_iv.isDouble());
        dropout = dropout_iv.toDouble();
    }

    // bool train
    int train = lstm2_layer->getTrain();
    if (nn_compiler::ir::isDefaultValue(train)) {
        auto train_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(train_iv.isInt());
        train = train_iv.toInt();
    }

    // bool bidirectional
    int bidirectional = lstm2_layer->getBidirectional();
    if (nn_compiler::ir::isDefaultValue(bidirectional)) {
        auto bidirectional_iv = stream_executor.findBlob(in_stensor_id[in_id++]).second;
        assert(bidirectional_iv.isInt());
        bidirectional = bidirectional_iv.toInt();
    }

    auto param_vector = lstm2_layer->getParamVector();

    auto pos = std::unique(out_stensor_id.begin(), out_stensor_id.end());
    out_stensor_id.erase(pos, out_stensor_id.end());
    assert(out_stensor_id.size() == 3);

    if (!input.is_contiguous()) input = input.contiguous();
    void *in_dev, *hx_dev, *out_dev, *wei_dev, *cx_dev, *workspace_dev, *hy_dev, *cy_dev;
    auto lstm_tensor = stream_executor.getMiopenLstmTensors();
    lstm_tensor.input_tensors_->clear();
    lstm_tensor.output_tensors_->clear();

    int batch_size = 1;
    int in_dim = input.dim();
    int input_size = input.size(in_dim - 1);
    int seq_len = input.size(in_dim - 2);
    int bidirectional_int = static_cast<bool>(bidirectional) ? 2 : 1;

    int hx_dim = hx_list_tensor_vector[0].dim();
    int hidden_size = hx_list_tensor_vector[0].size(hx_dim - 1);

    std::vector<int> in_len({batch_size, input_size});
    std::vector<int> hid_len({bidirectional_int * (int)(num_layers), hidden_size});
    std::vector<int> out_len({bidirectional_int * hidden_size});

    int dims = 2;
    for (int i = 0; i < seq_len; i++) {
        std::array<int, 2> in_lens = {in_len[0], in_len.back()};
        miopenCreateTensorDescriptor(lstm_tensor.input_tensor_);
        miopenSetTensorDescriptor(*(lstm_tensor.input_tensor_), miopenHalf, dims, in_lens.data(), nullptr);
        lstm_tensor.input_tensors_->push_back(*(lstm_tensor.input_tensor_));

        std::array<int, 2> out_lens = {{in_len[0], out_len[0]}};
        miopenCreateTensorDescriptor(lstm_tensor.output_tensor_);
        miopenSetTensorDescriptor(*(lstm_tensor.output_tensor_), miopenHalf, dims, out_lens.data(), nullptr);
        lstm_tensor.output_tensors_->push_back(*(lstm_tensor.output_tensor_));
    }
    std::array<int, 3> hid_lens = {{hid_len[0], in_len[0], hid_len[1]}};
    miopenSetTensorDescriptor(*(lstm_tensor.hidden_tensor_), miopenHalf, 3, hid_lens.data(), nullptr);

    miopenRNNMode_t mode = miopenRNNMode_t::miopenLSTM;
    miopenRNNBiasMode_t biasMode = static_cast<bool>(has_biases) ? miopenRNNwithBias : miopenRNNNoBias;
    miopenRNNDirectionMode_t directionMode = bidirectional_int == 2 ? miopenRNNbidirection : miopenRNNunidirection;
    miopenRNNInputMode_t inMode = miopenRNNlinear;
    miopenRNNAlgo_t algo = miopenRNNdefault;

    miopenSetRNNDescriptor(stream_executor.getMIOpenRNNDesc(), hidden_size, num_layers, inMode, directionMode, mode,
                           biasMode, algo, miopenHalf);
    miopenGetRNNParamsDescriptor(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(),
                                 *(lstm_tensor.input_tensor_), *(lstm_tensor.weight_tensor_), miopenHalf);
    size_t workspace_size;

    miopenTensorDescriptor_t* inputs_ptr = (*(lstm_tensor.input_tensors_)).data();
    miopenGetRNNWorkspaceSize(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(), seq_len,
                              inputs_ptr, &workspace_size);
    auto workspace = at::empty(workspace_size, input.options().dtype(at::kByte));

    in_dev = input.data_ptr();
    auto arranged_weight = lstm2_layer->getArrangedWeight();
    wei_dev = arranged_weight.data_ptr();
    hx_dev = hx_list_tensor_vector[0].data_ptr();
    cx_dev = hx_list_tensor_vector[1].data_ptr();
    workspace_dev = workspace.data_ptr();

    miopenTensorDescriptor_t* outputs_ptr = (*(lstm_tensor.output_tensors_)).data();
    if (stream_executor.getModelType() == "GNMT" &&
        stream_executor.findBlob(out_stensor_id[0]).first != ir::DataType::UNDEFINED && seq_len == 1) {
        out_dev = stream_executor.findBlob(out_stensor_id[0]).second.toTensor().data_ptr();
        hy_dev = stream_executor.findBlob(out_stensor_id[1]).second.toTensor().data_ptr();
        cy_dev = stream_executor.findBlob(out_stensor_id[2]).second.toTensor().data_ptr();
        miopenRNNForwardInference(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(), seq_len,
                                  inputs_ptr, in_dev, *(lstm_tensor.hidden_tensor_), hx_dev,
                                  *(lstm_tensor.hidden_tensor_), cx_dev, *(lstm_tensor.weight_tensor_), wei_dev,
                                  outputs_ptr, out_dev, *(lstm_tensor.hidden_tensor_), hy_dev,
                                  *(lstm_tensor.hidden_tensor_), cy_dev, workspace_dev, workspace_size);
    } else {
        auto output = at::empty({seq_len, out_len[0]}, input.options());
        out_dev = output.data_ptr();
        at::Tensor hy, cy;

        size_t hc_y_size = hid_lens[0] * hid_lens[1] * hid_lens[2];
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        hy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);
        cy = at::empty({hid_lens[0], hid_lens[1], hid_lens[2]}, options);

        hy_dev = hy.data_ptr();
        cy_dev = cy.data_ptr();

        miopenRNNForwardInference(stream_executor.getMIOpenHandle(), stream_executor.getMIOpenRNNDesc(), seq_len,
                                  inputs_ptr, in_dev, *(lstm_tensor.hidden_tensor_), hx_dev,
                                  *(lstm_tensor.hidden_tensor_), cx_dev, *(lstm_tensor.weight_tensor_), wei_dev,
                                  outputs_ptr, out_dev, *(lstm_tensor.hidden_tensor_), hy_dev,
                                  *(lstm_tensor.hidden_tensor_), cy_dev, workspace_dev, workspace_size);

        stream_executor.updateBlob(out_stensor_id[0], DataType::TENSOR, tensorToIValue(output));
        stream_executor.updateBlob(out_stensor_id[1], DataType::TENSOR, tensorToIValue(hy));
        stream_executor.updateBlob(out_stensor_id[2], DataType::TENSOR, tensorToIValue(cy));
    }
}

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
