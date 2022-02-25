#include <string>
#include <utility>

#include "compiler/include/common/log.hpp"
#include "compiler/include/frontend/optimizer/swap_matmul_inputs.h"

#include "new_ir/include/layers/aten_add_layer.h"
#include "new_ir/include/layers/aten_transpose_layer.h"
#include "new_ir/include/layers/pim_general_layers.h"
#include "new_ir/include/layers/prim_constant_layer.h"

#include "new_ir/include/utils/graph_search.h"
#include "new_ir/include/utils/graph_util.h"

namespace nn_compiler {

namespace frontend {

SwapMatmulInputs::SwapMatmulInputs() {
}

bool SwapMatmulInputs::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        if (layer->getType() == "aten::matmul") {
            auto predecessors = ir::searchPredecessor(layer, graph);
            if (predecessors.size() == 2 && predecessors[1]->getType() == "prim::Constant") {
                auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(predecessors[1]);
                auto dtensor = constant_layer->getAttr();
                std::vector<int> shape;
                auto b = dtensor->getTensorShape().getBatch();
                auto c = dtensor->getTensorShape().getChannel();
                auto h = dtensor->getTensorShape().getHeight();
                auto w = dtensor->getTensorShape().getWidth();
                if (b != 0) shape.push_back(b);
                if (c != 0) shape.push_back(c);
                if (h != 0) shape.push_back(h);
                if (w != 0) shape.push_back(w);

                if (shape.size() == 2) {
                    layers_.push_back(layer);
                }
            }
        }
    }
    return (layers_.size() != 0);
}

void SwapMatmulInputs::run(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    Log::FE::I() << "SwapMatmulInputs::run is called.";
    auto graph = model->getGraphs()[0];
    for (auto layer : layers_) {
        auto predecessors = ir::searchPredecessor(layer, graph);
        assert(predecessors.size() == 2);
        // create transpose layer and insert for matmul's input
        auto transpose_layer_for_input = std::make_shared<nn_compiler::ir::AtenTransposeLayer>("", "aten::transpose");
        transpose_layer_for_input->setName("aten::transpose_" + std::to_string(transpose_layer_for_input->getID()));
        // transpose with height and width
        transpose_layer_for_input->setDim0(-2);
        transpose_layer_for_input->setDim1(-1);
        // insert right after predecessor. However, can we insert it just before matmul?
        graph->addLayer2pos(transpose_layer_for_input, graph->getLayerPos(predecessors[0]));
        auto in_ids = layer->getInSTensorID();
        transpose_layer_for_input->addInSTensorID(in_ids[0]);
        auto new_stensor1 = std::make_shared<nn_compiler::ir::TSSTensor>();
        model->addTSSTensor(std::make_pair(new_stensor1->getID(), new_stensor1));
        new_stensor1->setFeaturemapType(model->getTSSTensors()[in_ids[0]]->getFeaturemapType());
        new_stensor1->setReprType(model->getTSSTensors()[in_ids[0]]->getReprType());
        transpose_layer_for_input->addOutSTensorID(new_stensor1->getID());
        layer->renewInSTensorID(0, new_stensor1->getID());

        // Transpose weight data
        transpose_for_constant(predecessors[1]);

        // swap input order of matmul
        auto in_stensor_ids = layer->getInSTensorID();
        assert(in_stensor_ids.size() == 2);
        layer->renewInSTensorID(0, in_stensor_ids[1]);
        layer->renewInSTensorID(1, in_stensor_ids[0]);

        // create transpose layer and insert for matmul's output
        auto transpose_layer_for_output = std::make_shared<nn_compiler::ir::AtenTransposeLayer>("", "aten::transpose");
        transpose_layer_for_output->setName("aten::transpose_" + std::to_string(transpose_layer_for_output->getID()));
        // transpose with height and width
        transpose_layer_for_output->setDim0(-2);
        transpose_layer_for_output->setDim1(-1);
        // insert right after matmul.
        graph->addLayer2pos(transpose_layer_for_output, graph->getLayerPos(layer));
        auto out_ids = layer->getOutSTensorID();
        auto new_stensor2 = std::make_shared<nn_compiler::ir::TSSTensor>();
        new_stensor2->setFeaturemapType(model->getTSSTensors()[out_ids[0]]->getFeaturemapType());
        new_stensor2->setReprType(model->getTSSTensors()[out_ids[0]]->getReprType());
        model->addTSSTensor(std::make_pair(new_stensor2->getID(), new_stensor2));
        layer->setOutSTensorID({new_stensor2->getID()});
        transpose_layer_for_output->addInSTensorID(layer->getOutSTensorID()[0]);
        transpose_layer_for_output->setOutSTensorID(out_ids);
    }
}

void SwapMatmulInputs::transpose_for_constant(std::shared_ptr<nn_compiler::ir::NNLayer>& layer) {
    auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(layer);
    auto dtensor = constant_layer->getAttr();
    auto height = dtensor->getTensorShape().getHeight();
    auto width = dtensor->getTensorShape().getWidth();
    auto matrix = constant_parser_.parse<half_float::half>(dtensor);

    std::vector<half_float::half> matrix_t_flat;
    for (auto i = 0; i < width; i ++) {
        for (auto j = 0; j < height; j ++) {
            matrix_t_flat.push_back(matrix[j][i]);
        }
    }

    dtensor->setData(matrix_t_flat.data(), matrix_t_flat.size() * sizeof(half_float::half));
    dtensor->setTensorShape(nn_compiler::ir::STensor(0, 0, width, height));
    dtensor->setStride({0, 0, 0, 0});  // changes at here make dtensor always contiguous
    constant_layer->setAttr(dtensor);
}

}  // namespace frontend
}  // namespace nn_compiler
