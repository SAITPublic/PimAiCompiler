#include "ir/include/utils/graph_print.h"

#include "ir/include/layers/prim_if_layer.h"
#include "ir/include/layers/prim_loop_layer.h"

#include "ir/include/types.h"

namespace nn_compiler {
namespace ir {

void printGraphModel(std::unique_ptr<ir::NNModel>& nn_model) {
    std::string tab = "    ";
    auto graphs = nn_model->getGraphs();
    for (auto graph : graphs) {
        auto gin_ids = graph->getGraphInTensorID();
        std::string network_line;
        network_line.clear();
        network_line.append(graph->getName() + " (");
        for (auto iid : gin_ids) {
            network_line = network_line.append(
                std::to_string(iid) + " : " +
                nn_model->getTSSTensors()[iid]->getReprType() + ", ");
        }
        network_line.append(") :");
        Log::IR::I() << network_line;

        std::string layer_line;
        for (auto layer : graph->getLayers()) {
            layer_line.clear();
            std::string attr = "";
            if (layer->getType() == LayerType::PRIMIF) {
                attr =
                    "[ThenNet = " +
                    std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(
                        layer)
                        ->getThenNet() +
                    ", ElseNet = " +
                    std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(
                        layer)
                        ->getElseNet() +
                    "]";
            }
            if (layer->getType() == LayerType::PRIMLOOP) {
                attr = "[BodyNet = " +
                    std::dynamic_pointer_cast<
                        nn_compiler::ir::PrimLoopLayer>(layer)
                        ->getBodyNet() +
                    "]";
            }
            layer_line.append(tab);
            auto in_ids = layer->getInSTensorID();
            auto out_ids = layer->getOutSTensorID();
            for (auto oid : out_ids) {
                layer_line.append(
                    std::to_string(oid) + " : " +
                    nn_model->getTSSTensors()[oid]->getReprType() + ", ");
            }
            layer_line.append(" = " + convertLayerTypeToString(layer->getType()) + attr + " (");
            for (auto iid : in_ids) {
                layer_line.append(std::to_string(iid) + ", ");
            }
            layer_line.append(")");
            Log::IR::I() << layer_line;
        }

        auto gout_ids = graph->getGraphOutTensorID();
        std::string return_line;
        return_line.clear();
        return_line.append(tab + "return (");
        for (auto oid : gout_ids) {
            return_line.append(std::to_string(oid) +  ", ");
        }
        return_line.append(") \n\n");
        Log::IR::I() << return_line;
    }
}

} // namespace ir
} // namespace nn_compiler
