#include "middlend/optimizer/stream_execution/update_layer_id.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace middlend
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
    for (auto layer : graph->getLayers()) {
        if (layer->getType() == ir::LayerType::MULTISTREAM) {
            auto muti_stream_layer = std::static_pointer_cast<nn_compiler::ir::MultiStreamLayer>(layer);
            auto execution_layers = muti_stream_layer->getLayers();
            for (auto execution_layer : execution_layers) {
                setLayerRelations(execution_layer, model);
            }
        } else {
            setLayerRelations(layer, model);
        }
    }
}

void UpdateLayerId::setLayerRelations(std::shared_ptr<ir::NNLayer>& layer,
                                      std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto layer_relations_map = model->getLayerRelationShips();
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

}  // namespace middlend
}  // namespace nn_compiler
