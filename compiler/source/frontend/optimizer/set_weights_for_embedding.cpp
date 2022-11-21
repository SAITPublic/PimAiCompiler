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

#include <torch/script.h>

#include "frontend/optimizer/set_weights_for_embedding.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
SetWeightsForEmbedding::SetWeightsForEmbedding() {}

bool SetWeightsForEmbedding::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model)
{
    auto graphs = graph_model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::ATENEMBEDDING) {
                layers_.push_back(layer);
            }
        }
    }

    return (layers_.size() != 0);
}

void SetWeightsForEmbedding::run(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model)
{
    DLOG(INFO) << "SetWeightsForEmbedding::run is called.";
    auto graph = graph_model->getGraphs()[0];

    for (auto layer : layers_) {
        auto cur_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenEmbeddingLayer>(layer);
        auto predecessors = ir::utils::searchPredecessor(layer, graph_model);
        assert(predecessors.size() > 0);
        if (predecessors[0]->getType() == nn_compiler::ir::LayerType::PRIMCONSTANT) {
            auto constant_g_layer = predecessors[0];
            auto idx = 0;

            auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(constant_g_layer);
            auto d_tensor = constant_layer->getAttr();
            auto tensor_shape = d_tensor->getTensorShape().getDims();
            assert(tensor_shape.size() == 4);  // n, c, h, w
            auto height = tensor_shape[2], width = tensor_shape[3];
            std::vector<int> weights_shape = {(height, width)};
            cur_layer->setWeightsShape(weights_shape);
            std::vector<at::Tensor> weights;
            auto matrix = constant_parser_.parse<half_float::half>(d_tensor);
            std::vector<half_float::half> matrix_t_flat;
            for (auto i = 0; i < height; i++) {
                for (auto j = 0; j < width; j++) {
                    matrix_t_flat.push_back(matrix[i][j]);
                }
            }
            int count = 0;
            for (auto h = 0; h < height; h++) {
                std::vector<half_float::half> data;
                for (auto w = 0; w < width; w++) {
                    data.push_back(matrix_t_flat[count++]);
                }
                auto data_tensor_cpu =
                    torch::from_blob(data.data(), {1, 1, width}, at::TensorOptions().dtype(torch::kFloat16));
                auto data_tensor = std::move(data_tensor_cpu.cuda());
                weights.push_back(data_tensor);
            }
            cur_layer->setWeights(weights);

            auto successors_of_constant = ir::utils::searchMapSuccessor(constant_g_layer, graph_model);
            if (successors_of_constant.size() == 1) {  // a constant only for embedding weight
                cur_layer->deleteInSTensorID(idx);
                graph->deleteLayer(constant_layer->getID());
            }
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
