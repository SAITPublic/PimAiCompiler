#include "common/include/common.hpp"
#include "compiler/include/common/log.hpp"

#include "compiler/include/frontend/optimizer/cat_labeling.h"
#include "new_ir/include/layers/aten_cat_layer.h"
#include "new_ir/include/layers/aten_lstm1_layer.h"

#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/nn_model.h"
#include "new_ir/include/nn_network.h"

#include "new_ir/include/types.h"

#include <algorithm>
#include <set>
#include <vector>

namespace nn_compiler
{
namespace frontend
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
    auto out_layers_id = layer->getOutSTensorID();
    for (auto out_id : out_layers_id) {
        for (auto check_layer : graph->getLayers()) {
            if (check_layer->getName() == layer->getName() && check_layer->getType() == targetLayerType &&
                std::find(res.begin(), res.end(), check_layer->getID()) == res.end()) {
                res.emplace_back(check_layer->getID());
            }
            getOffspring(res, graph, check_layer, targetLayerType, next_level);
        }
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
            if (graph->getLayer(cat_id)) {
                auto cat_layer = std::dynamic_pointer_cast<ir::AtenCatLayer>(graph->getLayer(cat_id));
                auto cat_in_layer_id = cat_layer->getInSTensorID()[0];
                for (auto cat_in_layer : graph->getLayers()) {
                    if (cat_in_layer->getID() == cat_in_layer_id) {
                        cat_layer->setMemLayerId(cat_in_layer->getID());
                        break;
                    }
                }
                break;
            }
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler