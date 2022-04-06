#include "compiler/include/frontend/optimizer/remove_constant_layers.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
RemoveConstantLayers::RemoveConstantLayers() {}

bool RemoveConstantLayers::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMCONSTANT) {
                auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(layer);
                if (constant_layer->getToRemove()) {
                    remove_layers_.push_back(layer);
                }
            } else if (layer->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE) {
                auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
                if (variable_layer->getToRemove()) {
                    remove_layers_.push_back(layer);
                }
            }
        }
    }

    return (remove_layers_.size() != 0);
}

void RemoveConstantLayers::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "RemoveConstantLayers::run is called.";

    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : remove_layers_) {
        auto successors = ir::utils::searchSuccessors(layer, graph);
        // There is always only one output from prim::Constant or prim::Variable,
        // and the output could be used by multiple layers.
        auto out_stensor_id = layer->getOutSTensorID()[0];
        for (auto successor : successors) {
            for (auto inID : successor.second) {
                (successor.first)->deleteInSTensorID(inID);
            }
        }
        graph->deleteLayer(layer->getID());
        graph->deleteSTensor(out_stensor_id);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
