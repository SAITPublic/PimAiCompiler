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

#include "frontend/optimizer/swap_addmm_inputs.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
SwapAddmmInputs::SwapAddmmInputs() {}

bool SwapAddmmInputs::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        if (layer->getType() == nn_compiler::ir::LayerType::ATENADDMM) {
            auto predecessors = ir::utils::searchPredecessor(layer, model);
            // at least: bias, input, weight.
            if (predecessors.size() >= 3 && predecessors[2]->getType() == nn_compiler::ir::LayerType::PRIMCONSTANT) {
                layers_.push_back(layer);
            }
        }
    }
    return (layers_.size() != 0);
}

void SwapAddmmInputs::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "SwapAddmmInputs::run is called.";
    auto graph = model->getGraphs()[0];
    for (auto layer : layers_) {
        auto predecessors = ir::utils::searchPredecessor(layer, model);

        // create transpose layer and insert for addmm's input
        auto transpose_layer_for_input =
            std::make_shared<nn_compiler::ir::AtenTransposeLayer>("", nn_compiler::ir::LayerType::ATENTRANSPOSE);
        transpose_layer_for_input->setName("aten::transpose_" + std::to_string(transpose_layer_for_input->getID()));
        // transpose with height and width
        transpose_layer_for_input->setDim0(-2);
        transpose_layer_for_input->setDim1(-1);
        // insert transpose layer before addmm.
        graph->addLayer2pos(transpose_layer_for_input, graph->getLayerPos(layer) - 1);
        auto in_id = layer->getInSTensorID()[1];
        transpose_layer_for_input->addInSTensorID(in_id);

        model->updateLayerRelationShips(in_id, layer, transpose_layer_for_input);

        auto new_stensor1 = std::make_shared<nn_compiler::ir::STensor>();
        auto idx1 = new_stensor1->getID();
        model->addSTensor(std::make_pair(idx1, new_stensor1));
        new_stensor1->setFeaturemapType(model->getSTensors()[in_id]->getFeaturemapType());
        new_stensor1->setReprType(model->getSTensors()[in_id]->getReprType());

        transpose_layer_for_input->addOutSTensorID(idx1);
        model->addLayerRelationShips(idx1, transpose_layer_for_input);
        model->addLayerRelationShips(idx1, layer);
        layer->renewInSTensorID(1, idx1);

        // swap input order of addmm
        auto in_stensor_ids = layer->getInSTensorID();
        layer->renewInSTensorID(1, in_stensor_ids[2]);
        layer->renewInSTensorID(2, in_stensor_ids[1]);

        // create transpose layer and insert for addmm's output
        auto transpose_layer_for_output =
            std::make_shared<nn_compiler::ir::AtenTransposeLayer>("", nn_compiler::ir::LayerType::ATENTRANSPOSE);
        transpose_layer_for_output->setName("aten::transpose_" + std::to_string(transpose_layer_for_output->getID()));
        // transpose with height and width
        transpose_layer_for_output->setDim0(-2);
        transpose_layer_for_output->setDim1(-1);
        // insert right after addmm.
        graph->addLayer2pos(transpose_layer_for_output, graph->getLayerPos(layer));
        auto out_ids = layer->getOutSTensorID();

        model->updateLayerRelationShips(out_ids[0], layer, transpose_layer_for_output);

        auto new_stensor2 = std::make_shared<nn_compiler::ir::STensor>();
        auto idx2 = new_stensor2->getID();
        new_stensor2->setFeaturemapType(model->getSTensors()[out_ids[0]]->getFeaturemapType());
        new_stensor2->setReprType(model->getSTensors()[out_ids[0]]->getReprType());

        model->addSTensor(std::make_pair(idx2, new_stensor2));
        layer->setOutSTensorID({idx2});
        transpose_layer_for_output->addInSTensorID(idx2);
        model->addLayerRelationShips(idx2, transpose_layer_for_output);
        model->addLayerRelationShips(idx2, layer);
        transpose_layer_for_output->setOutSTensorID(out_ids);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
