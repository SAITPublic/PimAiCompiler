#include "compiler/include/frontend/optimizer/remove_dropout_layers.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{

RemoveDropoutLayers::RemoveDropoutLayers() {}

bool RemoveDropoutLayers::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::ATENDROPOUT) {
                remove_layers_.push_back(layer);
            }
        }
    }

    return (remove_layers_.size() != 0);
}

void RemoveDropoutLayers::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "RemoveDropoutLayers::run is called.";

    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : remove_layers_) {
        /* aten::dropout:
            %input1.1 : Tensor = aten::dropout(%input0.1, %131, %132),
            where %input0.1 is input, %131 and %132 are from constants.
            There is always only one input, one output for aten::dropout.
        */
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
