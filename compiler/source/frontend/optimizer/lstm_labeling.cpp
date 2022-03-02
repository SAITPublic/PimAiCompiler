#include "common/include/common.hpp"
#include "compiler/include/common/log.hpp"

#include "compiler/include/frontend/optimizer/lstm_labeling.h"
#include "new_ir/include/layers/aten_lstm1_layer.h"

#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/nn_model.h"
#include "new_ir/include/nn_network.h"

#include "new_ir/include/types.h"

#include <set>
#include <vector>

namespace nn_compiler
{

namespace frontend
{

LstmLabeling::LstmLabeling() {}

bool LstmLabeling::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == ir::LayerType::ATENLSTM1) {
                aten_lstm1_layers_.push_back(layer);
            }
        }
    }
    return (aten_lstm1_layers_.size() != 0);
}

void FindLayerByOutIDs(std::shared_ptr<nn_compiler::ir::NNNetwork> nn_net,
                       std::shared_ptr<nn_compiler::ir::NNLayer> layer,
                       std::set<std::shared_ptr<nn_compiler::ir::NNLayer>> tuple_construct_layers,
                       ir::LayerType layer_label, int& to_tuple_num)
{
    auto lstm_out_layers_id = layer->getOutSTensorID();
    for (auto out_id : lstm_out_layers_id) {
        for (auto check_layer : nn_net->getLayers()) {
            if (check_layer->getName() != layer->getName()) {
                auto lstm_in_layers_id = check_layer->getInSTensorID();
                for (auto in_id : lstm_in_layers_id) {
                    if (out_id == in_id && check_layer->getType() == layer_label) {
                        tuple_construct_layers.insert(check_layer);
                        to_tuple_num++;
                    }
                }
            }
        }
    }
}

void SetLayerByOutIDs(std::shared_ptr<nn_compiler::ir::NNNetwork> nn_net,
                      std::shared_ptr<nn_compiler::ir::NNLayer> layer, ir::LayerType layer_label, int& custom_opt_number)
{
    auto lstm_out_layers_id = layer->getOutSTensorID();
    for (auto out_id : lstm_out_layers_id) {
        for (auto check_layer : nn_net->getLayers()) {
            if (check_layer->getName() != layer->getName()) {
                auto lstm_in_layers_id = check_layer->getInSTensorID();
                for (auto in_id : lstm_in_layers_id) {
                    if (out_id == in_id && check_layer->getType() == layer_label) {
                        auto lstm_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(check_layer);
                        lstm_layer->setMatchCustomOpt(true);
                        lstm_layer->setCustomOptNumber(custom_opt_number++);
                    }
                }
            }
        }
    }
}

void LstmLabeling::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    Log::FE::I() << "LstmLabeling::run is called.";
    int custom_opt_number = 0;

    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == ir::LayerType::ATENLSTM1) {
                std::set<std::shared_ptr<nn_compiler::ir::NNLayer>> tuple_construct_layers;
                int to_tuple_num = 0;
                FindLayerByOutIDs(graph, layer, tuple_construct_layers, ir::LayerType::PRIMTUPLECONSTRUCT, to_tuple_num);
                // two outputs from lstm1 node come to a single tuple_construct node.
                if (to_tuple_num == 2 && tuple_construct_layers.size() == 1) {
                    auto tuple_construct_layer = *tuple_construct_layers.begin();
                    SetLayerByOutIDs(graph, tuple_construct_layer, ir::LayerType::ATENAPPEND, custom_opt_number);
                }
            }
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
