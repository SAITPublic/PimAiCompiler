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

#include "frontend/optimizer/take_in_body_net.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
TakeInBodyNet::TakeInBodyNet() {}

void TakeInBodyNet::fitIfCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMIF) {
                prim_if_layers_.insert(prim_if_layers_.begin(), std::make_pair(layer, graph));
            }
        }
    }
    return;
}

void TakeInBodyNet::fitLoopCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMLOOP) {
                prim_loop_layers_.insert(prim_loop_layers_.begin(), std::make_pair(layer, graph));
            }
        }
    }
    return;
}

bool TakeInBodyNet::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    fitIfCondition(model);
    fitLoopCondition(model);
    return (prim_if_layers_.size() != 0 || prim_loop_layers_.size() != 0);
}

void TakeInBodyNet::take_in_if_body(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "Take in if body net start.";
    auto graphs = model->getGraphs();
    for (auto layer_pair : prim_if_layers_) {
        std::shared_ptr<nn_compiler::ir::NNGraph> main_graph = layer_pair.second;
        auto layer = layer_pair.first;
        auto prim_if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(layer);
        std::shared_ptr<nn_compiler::ir::NNGraph> then_net = nullptr, else_net = nullptr;
        for (auto graph : graphs) {
            if (graph->getName() == prim_if_layer->getThenNet()) {
                if (then_net != nullptr) {
                    DLOG(FATAL) << "duplicated network name occurs.";
                }
                then_net = graph;
            } else if (graph->getName() == prim_if_layer->getElseNet()) {
                if (else_net != nullptr) {
                    DLOG(FATAL) << "duplicated network name occurs.";
                }
                else_net = graph;
            }
        }
        if (then_net == nullptr || else_net == nullptr) {
            DLOG(FATAL) << (then_net == nullptr ? "then" : "else") << " branch of prim::If missed.";
        }
        std::vector<std::shared_ptr<nn_compiler::ir::NNGraph>> if_nets = {then_net, else_net};
        std::vector<uint32_t> if_layer_out_tensor_ids = layer->getOutSTensorID();
        std::vector<uint32_t> if_end_in_id;
        std::vector<uint32_t> if_layer_out_id;

        auto end_layer = std::make_shared<nn_compiler::ir::PrimEndIfLayer>("", nn_compiler::ir::LayerType::PRIMENDIF);
        end_layer->setName(convertLayerTypeToString(end_layer->getType()) + "_" + std::to_string(end_layer->getID()));
        end_layer->setOutSTensorID(if_layer_out_tensor_ids);

        for (auto idx : if_layer_out_tensor_ids) {
            model->updateLayerRelationShips(idx, layer, end_layer);
        }

        uint32_t layer_pos_in_main_graph = 0;
        // find if layer's pos in main graph, and insert current layers into if layer behind
        auto main_graph_layers = main_graph->getLayers();
        for (uint32_t idx = 0; idx < main_graph_layers.size(); idx++) {
            if (main_graph_layers[idx] == layer) {
                layer_pos_in_main_graph = idx;
                break;
            }
        }
        for (unsigned int i = 0; i < if_nets.size(); i++) {
            auto net = if_nets[i];
            auto subnet_outputs = net->getGraphOutTensorID();
            auto layers = net->getLayers();
            // note: when if then or else net has no layers, we add updateLayerRelationShips for net outputs by using id
            // and nullptr in model build.
            if (layers.size() == 0) {
                for (auto idx : subnet_outputs) {
                    if_end_in_id.push_back(idx);
                    model->updateLayerRelationShips(idx, nullptr, end_layer);
                }
            } else {
                for (auto idx : subnet_outputs) {
                    if_end_in_id.push_back(idx);
                    model->addLayerRelationShips(idx, end_layer);
                }
            }

            for (uint32_t i = 0; i < layers.size(); i++) {
                auto temp_layer = layers[i];
                main_graph->addLayer2pos(temp_layer, layer_pos_in_main_graph++);
            }
            if (i == 0) {
                // add a endif layer to mark the end of then net
                auto end_then_net_marker =
                    std::make_shared<nn_compiler::ir::PrimEndIfLayer>("", nn_compiler::ir::LayerType::PRIMENDIF);
                end_then_net_marker->setName(convertLayerTypeToString(end_then_net_marker->getType()) + "_" +
                                             std::to_string(end_then_net_marker->getID()));
                main_graph->addLayer2pos(end_then_net_marker, layer_pos_in_main_graph++);
            }
            model->removeGraph(net->getName());
        }
        layer->setOutSTensorID(if_layer_out_id);
        end_layer->setInSTensorID(if_end_in_id);
        main_graph->addLayer2pos(end_layer, layer_pos_in_main_graph++);
    }
}

void TakeInBodyNet::take_in_loop_body(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "Take in loop body net start.";
    auto graphs = model->getGraphs();
    // need find loop layer belongs to which net. when loop layer in if body net, after take_in_if_body()
    // executed, if body net removed, it need update the map(prim_loop_layers_)
    prim_loop_layers_.clear();
    fitLoopCondition(model);
    for (auto layer_pair : prim_loop_layers_) {
        std::shared_ptr<nn_compiler::ir::NNGraph> main_graph = layer_pair.second;
        auto layer = layer_pair.first;
        auto prim_loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(layer);
        std::shared_ptr<nn_compiler::ir::NNGraph> body_net = nullptr;
        for (auto graph : graphs) {
            if (graph->getName() == prim_loop_layer->getBodyNet()) {
                if (body_net != nullptr) {
                    DLOG(FATAL) << "duplicated network name occurs.";
                }
                body_net = graph;
            }
        }
        if (body_net == nullptr) {
            DLOG(FATAL) << " body block of prim::Loop missed.";
        }

        std::vector<uint32_t> loop_layer_out_tensor_ids = layer->getOutSTensorID();
        auto subnet_inputs = body_net->getGraphInTensorID();
        std::vector<uint32_t> new_loop_to_block_id;
        for (auto idx : loop_layer_out_tensor_ids) {
            auto id = getUniqueTensorId(model);
            auto shape_tensors = model->getSTensors();
            auto data_type = shape_tensors[idx]->getFeaturemapType();
            auto index_to_block_id_shape_tensor = shape_tensors[id];
            index_to_block_id_shape_tensor->setFeaturemapType(data_type);
            new_loop_to_block_id.push_back(id);
            model->deleteLayerRelationShips(idx, layer);
        }

        // find loop layer's pos in main graph, and insert current layers into loop layer behind

        uint32_t layer_pos_in_main_graph = 0;
        auto main_graph_layers = main_graph->getLayers();
        for (uint32_t idx = 0; idx < main_graph_layers.size(); idx++) {
            if (main_graph_layers[idx] == layer) {
                layer_pos_in_main_graph = idx;
                break;
            }
        }
        auto layers = body_net->getLayers();
        for (uint32_t i = 0; i < layers.size(); i++) {
            auto temp_layer = layers[i];

            if (i == 0) {
                auto index_to_block_id = getUniqueTensorId(model);
                auto index_to_block_id_shape_tensor = model->getSTensors()[index_to_block_id];
                index_to_block_id_shape_tensor->setFeaturemapType(nn_compiler::ir::DataType::UINT8);
                auto loop_index_layer = std::make_shared<nn_compiler::ir::PrimLoopIndexLayer>(
                    "", nn_compiler::ir::LayerType::PRIMLOOPINDEX);
                loop_index_layer->setName(convertLayerTypeToString(loop_index_layer->getType()) + "_" +
                                          std::to_string(loop_index_layer->getID()));

                loop_index_layer->addOutSTensorID(index_to_block_id);
                model->addLayerRelationShips(index_to_block_id, loop_index_layer);

                auto block_layer =
                    std::make_shared<nn_compiler::ir::PrimBlockLayer>("", nn_compiler::ir::LayerType::PRIMBLOCK);
                block_layer->setName(convertLayerTypeToString(block_layer->getType()) + "_" +
                                     std::to_string(block_layer->getID()));

                model->addLayerRelationShips(index_to_block_id, block_layer);
                block_layer->addInSTensorID(index_to_block_id);
                for (auto idx : new_loop_to_block_id) {
                    block_layer->addInSTensorID(idx);
                    model->addLayerRelationShips(idx, block_layer);
                }
                block_layer->setOutSTensorID(subnet_inputs);
                for (auto idx : subnet_inputs) {
                    model->addLayerRelationShips(idx, block_layer);
                }

                // if loop has no output, we need add edge to link block and loop layer
                if (new_loop_to_block_id.size() == 0) {
                    auto loop_to_block_id = getUniqueTensorId(model);
                    auto loop_to_block_id_shape_tensor = model->getSTensors()[loop_to_block_id];
                    loop_to_block_id_shape_tensor->setFeaturemapType(nn_compiler::ir::DataType::UINT8);
                    new_loop_to_block_id.push_back(loop_to_block_id);
                    block_layer->addInSTensorID(loop_to_block_id);
                    model->addLayerRelationShips(loop_to_block_id, block_layer);
                }

                for (auto idx : new_loop_to_block_id) {
                    model->addLayerRelationShips(idx, layer);
                }

                layer->setOutSTensorID(new_loop_to_block_id);

                main_graph->addLayer2pos(loop_index_layer, layer_pos_in_main_graph++);
                main_graph->addLayer2pos(block_layer, layer_pos_in_main_graph++);

                main_graph->addLayer2pos(temp_layer, layer_pos_in_main_graph++);
            } else if (i == layers.size() - 1) {
                main_graph->addLayer2pos(temp_layer, layer_pos_in_main_graph++);
                auto subnet_outputs = body_net->getGraphOutTensorID();
                if (subnet_outputs.size() != 0) {
                    auto end_layer = std::make_shared<nn_compiler::ir::PrimEndLoopLayer>(
                        "", nn_compiler::ir::LayerType::PRIMENDLOOP);
                    end_layer->setName(convertLayerTypeToString(end_layer->getType()) + "_" +
                                       std::to_string(end_layer->getID()));
                    end_layer->setInSTensorID(subnet_outputs);
                    for (auto idx : subnet_outputs) {
                        model->addLayerRelationShips(idx, end_layer);
                    }
                    end_layer->setOutSTensorID(loop_layer_out_tensor_ids);
                    for (auto idx : loop_layer_out_tensor_ids) {
                        model->addLayerRelationShips(idx, end_layer);
                    }
                    main_graph->addLayer2pos(end_layer, layer_pos_in_main_graph++);
                }

            } else {
                main_graph->addLayer2pos(temp_layer, layer_pos_in_main_graph++);
            }
        }
        model->removeGraph(body_net->getName());
    }
}

void TakeInBodyNet::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "TakeInBodyNet::run is called.";

    take_in_if_body(model);

    take_in_loop_body(model);
}

uint32_t TakeInBodyNet::getUniqueTensorId(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto shape_tensor = std::make_shared<nn_compiler::ir::STensor>();
    model->addSTensor(std::make_pair(shape_tensor->getID(), shape_tensor));
    return shape_tensor->getID();
}

}  // namespace frontend
}  // namespace nn_compiler
