#include "frontend/optimizer/remove_get_attr_layers.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
RemoveGetAttrLayers::RemoveGetAttrLayers() {}

bool RemoveGetAttrLayers::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMGETATTR) {
                remove_layers_.push_back(layer);
            }
        }
    }

    return (remove_layers_.size() != 0);
}

void RemoveGetAttrLayers::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "RemoveGetAttrLayers::run is called.";
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : remove_layers_) {
        // there is always one input and one output from prim::GetAttr
        auto old_stensor_id = layer->getOutSTensorID()[0];
        auto new_stensor_id = layer->getInSTensorID()[0];

        model->deleteLayerRelationShips(new_stensor_id, layer);

        auto predecessors = ir::utils::searchPredecessor(layer, model);
        auto successors = ir::utils::searchMapSuccessors(layer, model);
        for (auto successor : successors) {
            for (auto idx : successor.second) {
                auto success_layer = successor.first;
                success_layer->renewInSTensorID(idx, new_stensor_id);
                model->addLayerRelationShips(new_stensor_id, success_layer);
            }
        }
        graph->deleteLayer(layer->getID());
        graph->deleteSTensor(old_stensor_id);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
