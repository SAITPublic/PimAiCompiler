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

#include "frontend/optimizer/remove_constant_layers.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
RemoveConstantLayers::RemoveConstantLayers() {}

bool RemoveConstantLayers::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMCONSTANT) {
                auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(layer);
                if (constant_layer->getToRemove()) {
                    remove_layers_.push_back(layer);
                }
            } else if (layer->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE) {
                auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
                if (variable_layer->getToRemove()) {
                    remove_layers_.push_back(layer);
                }
            }
        }
    }

    return (remove_layers_.size() != 0);
}

void RemoveConstantLayers::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "RemoveConstantLayers::run is called.";

    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : remove_layers_) {
        auto successors = ir::utils::searchMapSuccessors(layer, model);
        // There is always only one output from prim::Constant or prim::Variable,
        // and the output could be used by multiple layers.
        for (auto successor : successors) {
            for (auto inID : successor.second) {
                (successor.first)->deleteInSTensorID(inID);
            }
        }
        graph->deleteLayer(layer->getID());
    }
}

}  // namespace frontend
}  // namespace nn_compiler
