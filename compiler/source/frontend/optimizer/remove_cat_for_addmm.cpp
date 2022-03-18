#include <string>

#include "compiler/include/frontend/optimizer/remove_cat_for_addmm.h"
#include "ir/include/utils/graph_search.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler {

namespace frontend {

RemoveCatForAddmm::RemoveCatForAddmm() {
}

bool RemoveCatForAddmm::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    auto graph = model->getGraphs()[0];
    for (auto layer : graph->getLayers()) {
        if (layer->getType() == nn_compiler::ir::LayerType::PRIMLISTCONSTRUCT) {
            auto predecessors = ir::searchPredecessor(layer, graph);
            if (predecessors.size() != 2) {
                continue;
            }
            auto successors = ir::searchSuccessorLayerOnly(layer, graph);
            if (successors.size() == 1 && successors[0]->getType() == nn_compiler::ir::LayerType::ATENCAT) {
                auto cat_layer = successors[0];
                auto successors_of_cat = ir::searchSuccessorLayerOnly(cat_layer, graph);
                if (successors_of_cat.size() == 1 &&
                        successors_of_cat[0]->getType() == nn_compiler::ir::LayerType::ATENADDMM) {
                    auto addmm_layer = successors_of_cat[0];
                    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>
                                layer_set = {layer, cat_layer, addmm_layer};
                    layers_.push_back(layer_set);
                }
            }
        }
    }

    return (layers_.size() != 0);
}

void RemoveCatForAddmm::run(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    DLOG(INFO) << "RemoveCatForAddmm::run is called.";

    auto graph = model->getGraphs()[0];

    for (auto layer_set : layers_) {
        auto addmm_layer = layer_set[2];
        auto addmm_predecessors = ir::searchPredecessor(addmm_layer, graph);
        auto old_weight = addmm_predecessors[2];
        auto new_weights = create_new_constants(
            std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(old_weight));
        graph->deleteLayer(old_weight->getID());
        model->deleteTSSTensor(old_weight->getOutSTensorID()[0]);
        reorganize_graph(model, new_weights);
    }
}

std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>
RemoveCatForAddmm::create_new_constants(std::shared_ptr<nn_compiler::ir::PrimConstantLayer> old_constant_layer) {
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> new_constants;

    auto origin_dtensor = old_constant_layer->getAttr();
    auto origin_data_vec = origin_dtensor->getData<float16>();
    auto stride = origin_dtensor->getStride();
    auto x_stride = stride[2], y_stride = stride[3];

    float16 origin_data[shape_of_matmul_weight_[0]][shape_of_matmul_weight_[1]];
    for (auto i = 0; i < shape_of_matmul_weight_[0]; i ++) {
        auto idx_in_vec = i + x_stride - 1;
        for (auto j = 0; j < shape_of_matmul_weight_[1]; j ++) {
            origin_data[i][j] = (*origin_data_vec)[idx_in_vec];
            idx_in_vec += y_stride;
        }
    }

    std::vector<float16> contiguous_data_vec;
    for (auto i = 0; i < shape_of_matmul_weight_[0]; i ++) {
        for (auto j = 0; j < shape_of_matmul_weight_[1]; j ++) {
            contiguous_data_vec.push_back(origin_data[i][j]);
        }
    }

    std::vector<std::vector<float16>> new_data;
    std::vector<nn_compiler::ir::STensor> data_shapes;
    int data_start = 0, data_amount = 0;
    for (auto shape_of_input : shape_of_inputs_) {
        data_amount += shape_of_input[1] * shape_of_matmul_weight_[1];
        std::vector<float16> new_vec{(contiguous_data_vec.begin() + data_start),
                                     (contiguous_data_vec.begin() + data_amount)};
        new_data.push_back(new_vec);
        data_shapes.push_back(nn_compiler::ir::STensor(0, 0, shape_of_input[1], shape_of_matmul_weight_[1]));
        data_start = data_amount;
    }

    for (auto idx = 0; idx < new_data.size(); idx++) {
        auto type = nn_compiler::ir::LayerType::PRIMCONSTANT;
        auto name = "splitted_" + std::to_string(idx) + "_" + old_constant_layer->getName();
        auto ntype = old_constant_layer->getNType();
        auto new_dtensor = std::make_shared<nn_compiler::ir::DTensor>();
        new_dtensor->setDataType(origin_dtensor->getDataType());
        float16 new_data_arr[new_data[idx].size()];
        std::copy(new_data[idx].begin(), new_data[idx].end(), new_data_arr);
        new_dtensor->setData(new_data_arr, new_data[idx].size() * sizeof(float16));
        new_dtensor->setTensorShape(data_shapes[idx]);
        new_dtensor->setBitWidth(16);

        auto new_prim_constant_layer = std::make_shared<nn_compiler::ir::PrimConstantLayer>(name, type);
        new_prim_constant_layer->setNType(ntype);
        new_prim_constant_layer->setAttr(new_dtensor);

        new_constants.push_back(new_prim_constant_layer);
    }

    return new_constants;
}

void RemoveCatForAddmm::reorganize_graph(std::unique_ptr<nn_compiler::ir::NNModel>& model,
                                         std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> new_constants) {
    auto graph = model->getGraphs()[0];
    for (auto layer_set : layers_) {
        auto list_construct_layer = layer_set[0], cat_layer = layer_set[1];
        graph->deleteLayer(list_construct_layer->getID());
        graph->deleteLayer(cat_layer->getID());

        auto addmm_layer = layer_set[2];
        auto block_inputs = ir::searchPredecessor(list_construct_layer, graph);
        auto block_outputs = ir::searchSuccessors(addmm_layer, graph);
        assert(block_inputs.size() == 2);

        // update first addmm Op
        auto input_layer1 = block_inputs[0];
        auto out_stensor1_id = input_layer1->getOutSTensorID()[0];
        addmm_layer->renewInSTensorID(1, out_stensor1_id);
        auto new_constant_layer1 = new_constants[0];
        graph->addLayer2pos(new_constant_layer1, graph->getLayerPos(addmm_layer) - 1);
        auto constant_out_stensor1 = std::make_shared<nn_compiler::ir::TSSTensor>();
        new_constant_layer1->addOutSTensorID(constant_out_stensor1->getID());
        model->addTSSTensor(std::make_pair(constant_out_stensor1->getID(), constant_out_stensor1));
        addmm_layer->renewInSTensorID(2, constant_out_stensor1->getID());

        // create second addmm Op
        auto new_addmm_layer = std::make_shared<nn_compiler::ir::AtenAddmmLayer>("", nn_compiler::ir::LayerType::ATENADDMM);
        new_addmm_layer->setName("splitted_" + addmm_layer->getName());
        graph->addLayer2pos(new_addmm_layer, graph->getLayerPos(addmm_layer));
        auto new_addmm_out_stensor = std::make_shared<nn_compiler::ir::TSSTensor>();
        new_addmm_layer->addOutSTensorID(new_addmm_out_stensor->getID());
        model->addTSSTensor(std::make_pair(new_addmm_out_stensor->getID(), new_addmm_out_stensor));

        // update second addmm Op
        new_addmm_layer->addInSTensorID(addmm_layer->getOutSTensorID()[0]);
        auto input_layer2 = block_inputs[1];
        auto out_stensor2_id = input_layer2->getOutSTensorID()[0];
        new_addmm_layer->addInSTensorID(out_stensor2_id);
        auto new_constant_layer2 = new_constants[1];
        graph->addLayer2pos(new_constant_layer2, graph->getLayerPos(new_addmm_layer) - 1);
        auto constant_out_stensor2 = std::make_shared<nn_compiler::ir::TSSTensor>();
        model->addTSSTensor(std::make_pair(constant_out_stensor2->getID(), constant_out_stensor2));
        new_constant_layer2->addOutSTensorID(constant_out_stensor2->getID());
        new_addmm_layer->addInSTensorID(constant_out_stensor2->getID());

        // update output of second addmm Op
        for (auto block_output : block_outputs) {
            for (auto idx : block_output.second) {
                block_output.first->renewInSTensorID(idx, new_addmm_out_stensor->getID());
            }
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
