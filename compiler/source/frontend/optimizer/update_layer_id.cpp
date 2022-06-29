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
    DLOG(INFO) << "Update layer id start.";
    uint32_t layer_id = 0;
    auto graph = model->getGraphs()[0];

    // update layer ID
    for (auto layer : graph->getLayers()) {
        layer->setID(layer_id++);
    }

    // update layer's pre_layer_ids & next_layer_ids
    auto layer_relations_map = model->getLayerRelationShips();
    for (auto layer : graph->getLayers()) {
        std::vector<uint32_t> pre_layer_ids, next_layer_ids;
        auto in_stensor_ids = layer->getInSTensorID();
        for (auto in_id : in_stensor_ids) {
            auto relation_layers = layer_relations_map[in_id];
            for (auto r_layer : relation_layers) {
                for (auto r_layer_out_id : r_layer->getOutSTensorID()) {
                    if (r_layer_out_id == in_id) {
                        pre_layer_ids.push_back(r_layer->getID());
                        break;
                    }
                }
            }
        }

        auto out_stensor_ids = layer->getOutSTensorID();

        for (auto out_id : out_stensor_ids) {
            auto relation_layers = layer_relations_map[out_id];
            for (auto r_layer : relation_layers) {
                for (auto r_layer_in_id : r_layer->getInSTensorID()) {
                    if (r_layer_in_id == out_id) {
                        next_layer_ids.push_back(r_layer->getID());
                        break;
                    }
                }
            }
        }

        layer->setPreLayerIDs(pre_layer_ids);
        layer->setNextLayerIDs(next_layer_ids);
    }
    // DLOG(FATAL)<<"";
}

}  // namespace frontend
}  // namespace nn_compiler
