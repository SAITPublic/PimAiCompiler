#include <algorithm>
#include <set>
#include <vector>

#include "middlend/optimizer/memory_allocation/cat_labeling.h"

namespace nn_compiler
{
namespace middlend
{
CatLabeling::CatLabeling() {}

bool CatLabeling::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        if (layer->getType() == ir::LayerType::ATENBMM) {
            getOffspring(target_cat_ids_bmm_, graph, layer, ir::LayerType::ATENCAT, 3);
        } else if (layer->getType() == ir::LayerType::ATENLSTM1 &&
                   std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer)->getMatchCustomOpt()) {
            getOffspring(target_cat_ids_lstm_, graph, layer, ir::LayerType::ATENCAT, 3);
        }
    }

    return (target_cat_ids_bmm_.size() != 0 && target_cat_ids_lstm_.size() != 0);
}

void CatLabeling::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graph = model->getGraphs()[0];
    std::vector<int64_t> target_cat_ids;

    for (auto cat_bmm : target_cat_ids_bmm_) {
        if (std::find(target_cat_ids_lstm_.begin(), target_cat_ids_lstm_.end(), cat_bmm) !=
            target_cat_ids_lstm_.end()) {
            target_cat_ids.emplace_back(cat_bmm);
        }
    }

    for (auto cat_id : target_cat_ids) {
        if (graph->getLayerByID(cat_id)) {
            auto cat_layer = std::dynamic_pointer_cast<ir::AtenCatLayer>(graph->getLayerByID(cat_id));
            auto cat_in_layer_id = cat_layer->getInSTensorID()[0];
            for (auto cat_in_layer : graph->getLayers()) {
                if (cat_in_layer->getID() == cat_in_layer_id) {
                    cat_layer->setMemLayerId(cat_in_layer->getID() * 10);
                    break;
                }
            }
        }
    }
}

void CatLabeling::getOffspring(std::vector<int64_t>& res, std::shared_ptr<nn_compiler::ir::NNNetwork> graph,
                               std::shared_ptr<nn_compiler::ir::NNLayer> layer, ir::LayerType targetLayerType,
                               int level)
{
    if (level == 0 || layer->getOutSTensorID().size() == 0) {
        return;
    }
    int next_level = level - 1;
    auto out_layers_id = layer->getNextLayerIDs();
    for (auto out_id : out_layers_id) {
        auto out_layer = graph->getLayerByID(out_id);
        if (out_layer->getType() == targetLayerType &&
            std::find(res.begin(), res.end(), out_layer->getID()) == res.end()) {
            res.emplace_back(out_layer->getID());
        }
        getOffspring(res, graph, out_layer, targetLayerType, next_level);
    }
}

}  // namespace middlend
}  // namespace nn_compiler
