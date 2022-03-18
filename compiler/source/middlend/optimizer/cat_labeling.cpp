#include <algorithm>
#include <set>
#include <vector>

#include "compiler/include/middlend/optimizer/cat_labeling.h"
#include "ir/include/layers/aten_cat_layer.h"
#include "ir/include/layers/aten_lstm1_layer.h"
#include "ir/include/nn_model.h"
#include "ir/include/nn_network.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace middlend
{
CatLabeling::CatLabeling() {}

bool CatLabeling::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) { return true; }

void getOffspring(std::vector<int64_t>& res, std::shared_ptr<nn_compiler::ir::NNNetwork> graph,
                  std::shared_ptr<nn_compiler::ir::NNLayer> layer, ir::LayerType targetLayerType, int level)
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

void CatLabeling::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    std::vector<int64_t> target_cat_ids_bmm;
    std::vector<int64_t> target_cat_ids_lstm;
    std::vector<int64_t> target_cat_ids;

    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == ir::LayerType::ATENBMM) {
                getOffspring(target_cat_ids_bmm, graph, layer, ir::LayerType::ATENCAT, 3);
            } else if (layer->getType() == ir::LayerType::ATENLSTM1 &&
                       std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer)->getMatchCustomOpt()) {
                getOffspring(target_cat_ids_lstm, graph, layer, ir::LayerType::ATENCAT, 3);
            }
        }
    }

    for (auto cat_bmm : target_cat_ids_bmm) {
        if (std::find(target_cat_ids_lstm.begin(), target_cat_ids_lstm.end(), cat_bmm) != target_cat_ids_lstm.end()) {
            target_cat_ids.emplace_back(cat_bmm);
        }
    }

    for (auto cat_id : target_cat_ids) {
        auto graphs = model->getGraphs();
        for (auto graph : graphs) {
            if (graph->getLayerByID(cat_id)) {
                auto cat_layer = std::dynamic_pointer_cast<ir::AtenCatLayer>(graph->getLayerByID(cat_id));
                auto cat_in_layer_id = cat_layer->getInSTensorID()[0];
                for (auto cat_in_layer : graph->getLayers()) {
                    if (cat_in_layer->getID() == cat_in_layer_id) {
                        cat_layer->setMemLayerId(cat_in_layer->getID() * 10);
                        break;
                    }
                }
                break;
            }
        }
    }
}

}  // namespace middlend
}  // namespace nn_compiler
