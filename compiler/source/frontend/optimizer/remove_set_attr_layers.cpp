#include "compiler/include/frontend/optimizer/remove_set_attr_layers.h"

#include "new_ir/include/utils/graph_util.h"

#include "ir/include/common/log.hpp"
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
            if (layer->getType() == "prim::SetAttr") {
                auto predecessors = ir::searchPredecessor(layer, graph);
                if (predecessors.size() == 2 && predecessors[0]->getType() == "prim::Variable") {
                    remove_layers_.push_back(layer);
                }
            }
        }
    }

    return (remove_layers_.size() != 0);
}

void RemoveSetAttrLayers::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    Log::FE::I() << "RemoveSetAttrLayers::run is called.";
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : remove_layers_) {
        auto predecessors = ir::searchPredecessor(layer, graph);
        assert(predecessors.size() == 2);
        auto variable_layer = predecessors[0];
        auto compute_layer = predecessors[1];

        // update successsors of prim::Variable layer
        auto old_stensor_id = variable_layer->getOutSTensorID()[0];
        auto new_stensor_id = compute_layer->getOutSTensorID()[0];
        auto successors = ir::searchSuccessors(variable_layer, graph);
        for (auto successor : successors) {
            for (auto idx : successor.second) {
                (successor.first)->renewInSTensorID(idx, new_stensor_id);
            }
        }

        graph->deleteLayer(layer->getID());
        graph->deleteLayer(variable_layer->getID());
        graph->deleteSTensor(old_stensor_id);
    }
}
}  // namespace frontend
}  // namespace nn_compiler
