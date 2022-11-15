/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "ir/include/utils/graph_print.h"
#include "ir/include/layers/all_layers.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace ir
{
namespace utils
{
void printGraphModel(std::unique_ptr<ir::NNModel>& nn_model)
{
    std::string tab = "    ";
    auto graphs = nn_model->getGraphs();
    for (auto graph : graphs) {
        auto gin_ids = graph->getGraphInTensorID();
        std::string network_line;
        network_line.clear();
        network_line.append(graph->getName() + " (");
        for (auto iid : gin_ids) {
            network_line =
                network_line.append(std::to_string(iid) + " : " + nn_model->getSTensors()[iid]->getReprType() + ", ");
        }
        network_line.append(") :");
        DLOG(INFO) << network_line;

        std::string layer_line;
        for (auto layer : graph->getLayers()) {
            layer_line.clear();
            std::string attr = "";
            if (layer->getType() == LayerType::PRIMIF) {
                attr = "[ThenNet = " + std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(layer)->getThenNet() +
                       ", ElseNet = " + std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(layer)->getElseNet() +
                       "]";
            }
            if (layer->getType() == LayerType::PRIMLOOP) {
                attr = "[BodyNet = " + std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(layer)->getBodyNet() +
                       "]";
            }
            layer_line.append(tab);
            auto in_ids = layer->getInSTensorID();
            auto out_ids = layer->getOutSTensorID();
            for (auto oid : out_ids) {
                layer_line.append(std::to_string(oid) + " : " + nn_model->getSTensors()[oid]->getReprType() + ", ");
            }
            layer_line.append(" = " + convertLayerTypeToString(layer->getType()) + attr + " (");
            for (auto iid : in_ids) {
                layer_line.append(std::to_string(iid) + ", ");
            }
            layer_line.append(")");
            DLOG(INFO) << layer_line;
        }

        auto gout_ids = graph->getGraphOutTensorID();
        std::string return_line;
        return_line.clear();
        return_line.append(tab + "return (");
        for (auto oid : gout_ids) {
            return_line.append(std::to_string(oid) + ", ");
        }
        return_line.append(") \n\n");
        DLOG(INFO) << return_line;
    }
}

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
