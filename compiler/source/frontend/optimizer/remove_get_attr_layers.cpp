#include <string>

#include "compiler/include/common/log.hpp"
#include "compiler/include/frontend/optimizer/remove_get_attr_layers.h"
#include "new_ir/include/layers/prim_get_attr_layer.h"
#include "new_ir/include/utils/graph_util.h"

namespace nn_compiler {

namespace frontend {

RemoveGetAttrLayers::RemoveGetAttrLayers() {
}

bool RemoveGetAttrLayers::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
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

void RemoveGetAttrLayers::run(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    Log::FE::I() << "RemoveGetAttrLayers::run is called.";
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : remove_layers_) {
        // there is always one input and one output from prim::GetAttr
        auto old_stensor_id = layer->getOutSTensorID()[0];
        auto new_stensor_id = layer->getInSTensorID()[0];
        auto successors = ir::searchSuccessors(layer, graph);
        for (auto successor : successors) {
            for (auto idx : successor.second) {
                (successor.first)->renewInSTensorID(idx, new_stensor_id);
            }
        }
        graph->deleteLayer(layer->getID());
        graph->deleteSTensor(old_stensor_id);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
