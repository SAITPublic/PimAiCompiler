#include "frontend/optimizer/remove_set_attr_layers.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
RemoveSetAttrLayers::RemoveSetAttrLayers() {}

bool RemoveSetAttrLayers::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMSETATTR) {
                auto predecessors = ir::utils::searchPredecessor(layer, model);
                if (predecessors.size() == 2 &&
                    predecessors[0]->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE) {
                    remove_layers_.push_back(layer);
                }
            }
        }
    }

    return (remove_layers_.size() != 0);
}

void RemoveSetAttrLayers::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "RemoveSetAttrLayers::run is called.";
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : remove_layers_) {
        auto predecessors = ir::utils::searchPredecessor(layer, model);
        if (predecessors.size() != 2) {
            // the first variable layer has been removed
            continue;
        }
        auto variable_layer = predecessors[0];
        auto compute_layer = predecessors[1];

        // update successsors of prim::Variable layer
        auto old_stensor_id = variable_layer->getOutSTensorID()[0];
        auto new_stensor_id = compute_layer->getOutSTensorID()[0];

        auto successors = ir::utils::searchMapSuccessors(variable_layer, model);
        for (auto successor : successors) {
            if ((successor.first)->getType() == ir::LayerType::PRIMGETATTR) {
                for (auto idx : successor.second) {
                    auto success_layer = successor.first;
                    success_layer->renewInSTensorID(idx, new_stensor_id);
                    model->updateLayerRelationShips(new_stensor_id, layer, success_layer);
                }
            }
        }

        graph->deleteLayer(layer->getID());
        graph->deleteLayer(variable_layer->getID());
        graph->deleteSTensor(old_stensor_id);
    }
}
}  // namespace frontend
}  // namespace nn_compiler
