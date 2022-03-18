#include <string>

#include "compiler/include/middlend/optimizer/control_layer_execution.h"
#include "ir/include/tensors/data_tensor.h"
#include "ir/include/types.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{

namespace middlend
{

ControlLayerExecution::ControlLayerExecution() {}

bool ControlLayerExecution::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == ir::LayerType::PRIMIF || layer->getType() == ir::LayerType::PRIMENDIF ||
                layer->getType() == ir::LayerType::PRIMLOOP || layer->getType() == ir::LayerType::PRIMENDLOOP) {
                control_layers_.push_back(layer);
            }
        }
    }
    return (control_layers_.size() != 0);
}

void ControlLayerExecution::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "ControlLayerExecution::run is called.";
    std::stack<std::pair<bool, std::shared_ptr<nn_compiler::ir::PrimIfLayer>>> if_layers;
    std::stack<std::shared_ptr<nn_compiler::ir::PrimEndIfLayer>> then_net_end_if_layers;
    std::stack<std::shared_ptr<nn_compiler::ir::PrimLoopLayer>> loop_layers;
    for (auto layer : control_layers_) {
        int layer_id = layer->getID();
        if (layer->getType() == ir::LayerType::PRIMIF) {
            auto prim_if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(layer);
            if_layers.push(std::make_pair(false, prim_if_layer));
            if_layers.push(std::make_pair(true, prim_if_layer));
        } else if (layer->getType() == ir::LayerType::PRIMENDIF) {
            bool then_net = if_layers.top().first;
            auto prim_if_layer = if_layers.top().second;
            auto prim_end_if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimEndIfLayer>(layer);
            prim_end_if_layer->setIfLayerId(prim_if_layer->getID());
            if (then_net) {
                prim_if_layer->setElseNetStartLayer(prim_end_if_layer->getID() + 1);
                prim_end_if_layer->setIsElseNet(false);
                then_net_end_if_layers.push(prim_end_if_layer);
            } else {
                auto then_net_end_if_layer = then_net_end_if_layers.top();
                then_net_end_if_layer->setGotoLayer(prim_end_if_layer->getID());
                prim_end_if_layer->setGotoLayer(prim_end_if_layer->getID() + 1);
                prim_end_if_layer->setIsElseNet(true);
                then_net_end_if_layers.pop();
            }
            if_layers.pop();
        } else if (layer->getType() == ir::LayerType::PRIMLOOP) {
            auto prim_loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(layer);
            loop_layers.push(prim_loop_layer);
        } else if (layer->getType() == ir::LayerType::PRIMENDLOOP) {
            auto prim_end_loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimEndLoopLayer>(layer);
            auto prim_loop_layer = loop_layers.top();
            prim_end_loop_layer->setGotoLayer(prim_loop_layer->getID());
            prim_loop_layer->setGotoLayer(prim_end_loop_layer->getID() + 1);
            loop_layers.pop();
        }
    }
}

}  // namespace middlend
}  // namespace nn_compiler
