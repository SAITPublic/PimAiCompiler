#include "frontend/optimizer/update_layer_id.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
UpdateLayerId::UpdateLayerId() {}

bool UpdateLayerId::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) { return true; }

void UpdateLayerId::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    uint32_t layer_id = 0;
    auto graph = model->getGraphs()[0];

    // update layer ID
    for (auto layer : graph->getLayers()) {
        layer->setID(layer_id++);
    }

    // update layer's pre_layer_ids & next_layer_ids
    for (auto layer : graph->getLayers()) {
        auto predecessors = nn_compiler::ir::utils::searchPredecessor(layer, graph);
        auto successors = nn_compiler::ir::utils::searchSuccessorLayerOnly(layer, graph);
        std::vector<uint32_t> pre_layer_ids, next_layer_ids;
        for (auto predecessor : predecessors) {
            pre_layer_ids.push_back(predecessor->getID());
        }
        for (auto successor : successors) {
            next_layer_ids.push_back(successor->getID());
        }
        layer->setPreLayerIDs(pre_layer_ids);
        layer->setNextLayerIDs(next_layer_ids);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
