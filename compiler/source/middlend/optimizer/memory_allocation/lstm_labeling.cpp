/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#include "middlend/optimizer/memory_allocation/lstm_labeling.h"

namespace nn_compiler
{
namespace middlend
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

void LstmLabeling::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "LstmLabeling::run is called.";
    int custom_opt_number = 0;

    auto graph = model->getGraphs()[0];
    for (auto layer : aten_lstm1_layers_) {
        std::set<std::shared_ptr<nn_compiler::ir::NNLayer>> tuple_construct_layers;
        int to_tuple_num = 0;
        auto out_layers_id = layer->getNextLayerIDs();
        for (auto out_id : out_layers_id) {
            auto out_layer = graph->getLayerByID(out_id);
            if (out_layer->getType() == ir::LayerType::PRIMTUPLECONSTRUCT) {
                tuple_construct_layers.insert(out_layer);
                to_tuple_num++;
            }
        }
        // two outputs from lstm1 node come to a single tuple_construct node.
        if (to_tuple_num == 2 && tuple_construct_layers.size() == 1) {
            auto tuple_construct_layer = *tuple_construct_layers.begin();
            auto tuple_construct_out_stensor_ids = tuple_construct_layer->getOutSTensorID();
            // this tuple will be sent to aten::append only.
            if (tuple_construct_out_stensor_ids.size() != 1) {
                continue;
            }
            auto tuple_construct_out_layers_ids = tuple_construct_layer->getNextLayerIDs();
            auto out_layer = graph->getLayerByID(tuple_construct_out_layers_ids[0]);
            if (out_layer->getType() == ir::LayerType::ATENAPPEND) {
                auto lstm_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer);
                lstm_layer->setMatchCustomOpt(true);
                lstm_layer->setCustomOptNumber(custom_opt_number++);
            }
        }
    }
}

}  // namespace middlend
}  // namespace nn_compiler
