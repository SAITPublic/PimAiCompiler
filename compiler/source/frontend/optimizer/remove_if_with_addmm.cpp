/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "frontend/optimizer/remove_if_with_addmm.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
RemoveIfWithAddmm::RemoveIfWithAddmm() {}

bool RemoveIfWithAddmm::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel> &model)
{
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();
    assert(layers.size() > 0);

    // body of prim::If are not identified by edge connections now (changes required from runtime side),
    // need to recognize by node order instead.
    auto pre_layer = layers[0];
    for (auto idx = 1; idx < layers.size() - 1; idx++) {
        auto cur_layer = layers[idx];
        auto next_layer = layers[idx + 1];
        if (pre_layer->getType() == nn_compiler::ir::LayerType::PRIMIF &&
            cur_layer->getType() == nn_compiler::ir::LayerType::ATENADDMM &&
            next_layer->getType() == nn_compiler::ir::LayerType::PRIMENDIF) {
            if_layer_idx_.push_back(idx - 1);
        }
        pre_layer = cur_layer;
    }

    return (if_layer_idx_.size() != 0);
}

void RemoveIfWithAddmm::run(std::unique_ptr<nn_compiler::ir::NNModel> &model)
{
    DLOG(INFO) << "RemoveIfWithAddmm::run is called.";

    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> delete_layers;
    assert(layers.size() > 0);

    /* order in the vector of layers is:
        -> prim::If -> aten::addmm -> prim::EndIf -> aten::matmul -> aten::add -> prim::EndIf ->
    */
    for (auto layer_idx : if_layer_idx_) {
        // remove the layer and its predecessors which are only computed for this if branch
        getDeleteLayers(model, layers[layer_idx - 1], delete_layers);  // only one input for prim::If

        // remove verbose layers
        delete_layers.push_back(layers[layer_idx]);      // prim::If
        delete_layers.push_back(layers[layer_idx + 2]);  // prim::EndIf of addmm
        delete_layers.push_back(layers[layer_idx + 3]);  // aten::matmul
        delete_layers.push_back(layers[layer_idx + 4]);  // aten::add
        delete_layers.push_back(layers[layer_idx + 5]);  // prim::EndIf of matmul + add

        // maintain connection
        auto addmm_layer = layers[layer_idx + 1];
        auto new_id = addmm_layer->getOutSTensorID()[0];  // always only one output from aten::addmm

        auto end_if_layer = layers[layer_idx + 5];  // prim::EndIf of matmul + add

        auto successors = ir::utils::searchMapSuccessors(end_if_layer, model);
        for (auto successor : successors) {
            auto in_ID_idx = successor.second;
            for (auto idx = 0; idx < in_ID_idx.size(); idx++) {
                auto success_layer = successor.first;
                success_layer->renewInSTensorID(in_ID_idx[idx], new_id);
                model->updateLayerRelationShips(new_id, end_if_layer, success_layer);
            }
        }
    }

    for (auto delete_layer : delete_layers) {
        graph->deleteLayer(delete_layer->getID());
    }
}

void RemoveIfWithAddmm::getDeleteLayers(std::unique_ptr<nn_compiler::ir::NNModel> &model,
                                        std::shared_ptr<nn_compiler::ir::NNLayer> layer,
                                        std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> &delete_layers)
{
    if ((ir::utils::searchMapSuccessors(layer, model)).size() == 1) {
        // the compute result of layer is only used by this if branch
        delete_layers.push_back(layer);
        auto predecessors = ir::utils::searchPredecessor(layer, model);
        for (auto predecessor : predecessors) {
            getDeleteLayers(model, predecessor, delete_layers);
        }
    }
}
}  // namespace frontend
}  // namespace nn_compiler
