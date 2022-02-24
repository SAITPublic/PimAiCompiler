#include "compiler/include/frontend/optimizer/set_weights_for_embedding.h"

#include "new_ir/include/utils/graph_search.h"
#include "new_ir/include/utils/graph_util.h"

namespace nn_compiler {

namespace frontend {

SetWeightsForEmbedding::SetWeightsForEmbedding() {
}

bool SetWeightsForEmbedding::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model) {
    auto graphs = graph_model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == "aten::embedding") {
                layers_.push_back(layer);
            }
        }
    }

    return (layers_.size() != 0);
}

void SetWeightsForEmbedding::run(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model) {
    DLOG(INFO) << "SetWeightsForEmbedding::run is called.";
    auto graph = graph_model->getGraphs()[0];

    for (auto layer : layers_) {
        auto cur_layer = std::dynamic_pointer_cast<nn_compiler::ir::AtenEmbeddingLayer>(layer);
        auto predecessors = ir::searchPredecessor(layer, graph);
        assert(predecessors.size() > 0);
        if (predecessors[0]->getType() == "prim::Constant") {
            auto constant_g_layer = predecessors[0];
            auto idx = 0;

            auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(predecessors[0]);
            auto d_tensor = constant_layer->getAttr();
            auto height = d_tensor->getTensorShape().getHeight();
            auto width = d_tensor->getTensorShape().getWidth();
            auto matrix = constant_parser_.parse<half_float::half>(d_tensor);
            std::vector<half_float::half> matrix_t_flat;
            for (auto i = 0; i < height; i ++) {
                for (auto j = 0; j < width; j ++) {
                    matrix_t_flat.push_back(matrix[i][j]);
                }
            }

            std::vector<float> weights;
            for (auto item : matrix_t_flat) {
                weights.push_back(static_cast<float>(item));
            }
            cur_layer->setWeights(weights);
            auto weights_stensor = d_tensor->getTensorShape();
            std::vector<int> weights_shape = {(weights_stensor.getHeight()), weights_stensor.getWidth()};
            cur_layer->setWeightsShape(weights_shape);

            auto successors_of_constant = ir::searchSuccessor(constant_g_layer, graph);
            if (successors_of_constant.size() == 1) {  // a constant only for embedding weight
                cur_layer->deleteInSTensorID(idx);
                graph->deleteLayer(constant_layer->getID());
                auto out_stensor_id = constant_layer->getOutSTensorID()[0];  // always one output stensor from constant
                graph->deleteSTensor(out_stensor_id);
            }
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
