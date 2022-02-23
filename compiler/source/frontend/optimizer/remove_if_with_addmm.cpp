#include <string>

#include "compiler/include/frontend/optimizer/remove_if_with_addmm.h"

#include "new_ir/include/layers/aten_addmm_layer.h"
#include "new_ir/include/utils/graph_search.h"
#include "new_ir/include/utils/graph_util.h"
#include "new_ir/include/layers/pim_general_layers.h"
#include "new_ir/include/layers/prim_if_layer.h"

namespace nn_compiler {

namespace frontend {

RemoveIfWithAddmm::RemoveIfWithAddmm() {
}

bool RemoveIfWithAddmm::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel> &model) {
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();
    assert(layers.size() > 0);

    // body of prim::If are not identified by edge connections now (changes required from runtime side),
    // need to recognize by node order instead.
    auto pre_layer = layers[0];
    for (auto idx = 1; idx < layers.size() - 1; idx++) {
        auto cur_layer = layers[idx];
        auto next_layer = layers[idx + 1];
        if (pre_layer->getType() == "prim::If" &&
                cur_layer->getType() == "aten::addmm" &&
                    next_layer->getType() == "prim::EndIf") {
            if_layer_idx_.push_back(idx - 1);
        }
        pre_layer = cur_layer;
    }

    return (if_layer_idx_.size() != 0);
}

void RemoveIfWithAddmm::run(std::unique_ptr<nn_compiler::ir::NNModel> &model) {
    DLOG(INFO) << "RemoveIfWithAddmm::run is called.";

    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> delete_layers;
    assert(layers.size() > 0);

    /* order in the vector of layers is: 
        -> prim::If -> aten::addmm -> prim::EndIf -> aten::matmul -> aten::add -> prim::EndIf ->
    */
    for (auto layer_idx : if_layer_idx_) {
        // remove the layer and its predecessors which are only computed for this if branch
        getDeleteLayers(graph, layers[layer_idx - 1], delete_layers);  // only one input for prim::If

        // remove verbose layers
        delete_layers.push_back(layers[layer_idx]);      // prim::If
        delete_layers.push_back(layers[layer_idx + 2]);  // prim::EndIf of addmm
        delete_layers.push_back(layers[layer_idx + 3]);  // aten::matmul
        delete_layers.push_back(layers[layer_idx + 4]);  // aten::add
        delete_layers.push_back(layers[layer_idx + 5]);  // prim::EndIf of matmul + add

        // maintain connection
        auto addmm_layer = layers[layer_idx + 1];
        auto new_id = addmm_layer->getOutSTensorID()[0];  // always only one output from aten::addmm

        auto end_if_layer = layers[layer_idx + 5];  // prim::EndIf of matmul + add
        auto successors = ir::searchSuccessors(end_if_layer, graph);
        for (auto successor : successors) {
            auto in_ID_idx = successor.second;
            for (auto idx = 0; idx < in_ID_idx.size(); idx++) {
                (successor.first)->renewInSTensorID(in_ID_idx[idx], new_id);
            }
        }
    }

    for (auto delete_layer : delete_layers) {
        graph->deleteLayer(delete_layer->getID());
    }
}

void RemoveIfWithAddmm::getDeleteLayers(std::shared_ptr<nn_compiler::ir::NNNetwork> graph,
                                        std::shared_ptr<nn_compiler::ir::NNLayer> layer,
                                        std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>& delete_layers) {
    if ((ir::searchSuccessors(layer, graph)).size()  == 1) {
        // the compute result of layer is only used by this if branch
        delete_layers.push_back(layer);
        auto predecessors = ir::searchPredecessor(layer, graph);
        for (auto predecessor : predecessors) {
            getDeleteLayers(graph, predecessor, delete_layers);
        }
    }
}
}  // namespace frontend
}  // namespace nn_compiler
