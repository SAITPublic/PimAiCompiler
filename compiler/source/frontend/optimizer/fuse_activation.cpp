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

#include "frontend/optimizer/fuse_activation.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
FuseActivation::FuseActivation() {}

bool FuseActivation::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto dependCheck = [&](const std::unique_ptr<ir::NNModel>& model,
                           const std::shared_ptr<nn_compiler::ir::NNLayer> predecessor,
                           const std::shared_ptr<nn_compiler::ir::NNLayer> successor) {
        std::string predecessor_type = convertLayerTypeToString(predecessor->getType());
        if (!this->feasibleHostType(predecessor_type)) {
            DLOG(INFO) << "failed to satisfy with fusion dependency";
            return false;
        }

        auto successors = ir::utils::searchMapSuccessor(predecessor, model);
        if (successors.size() > 1) {
            DLOG(INFO) << "failed to satisfy with fusion dependency";
            return false;
        }

        if (predecessor_type.compare("aten::transpose") == 0) {
            if (successor->getType() != nn_compiler::ir::LayerType::ATENADDMM) {
                DLOG(INFO) << "The predecessor of transpose layer is not addmm layer.";
                return false;
            }
            auto successors = ir::utils::searchMapSuccessor(successor, model);
            if (successors.size() > 1) {
                DLOG(INFO) << "failed to satisfy with fusion dependency";
                return false;
            }
        }
        return true;
    };
    // after take in body in net pass, there only one graph
    auto graph = model->getGraphs()[0];

    for (auto cur_layer : graph->getLayers()) {
        std::string type = convertLayerTypeToString(cur_layer->getType());
        if (feasibleParasiteType(type)) {
            auto predecessors = ir::utils::searchPredecessor(cur_layer, model);
            assert(predecessors.size() > 0);
            auto pre_of_predecessor = ir::utils::searchPredecessor(predecessors[0], model);
            if (predecessors.empty() || !dependCheck(model, predecessors[0], pre_of_predecessor[0])) {
                continue;
            }

            layers_.push_back(cur_layer);
        }
    }

    return (layers_.size() != 0);
}

void FuseActivation::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "FuseActivation::run is called.";
    auto graph = model->getGraphs()[0];

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_to_be_removed;
    for (auto cur_layer : layers_) {
        std::string type = convertLayerTypeToString(cur_layer->getType());
        if (feasibleParasiteType(type)) {
            auto predecessors = ir::utils::searchPredecessor(cur_layer, model);
            auto successors = ir::utils::searchPredecessor(predecessors[0], model);

            auto out_ids = cur_layer->getOutSTensorID();
            auto in_ids = cur_layer->getInSTensorID();
            CHECK_EQ(out_ids.size(), 1);
            CHECK_EQ(in_ids.size(), 1);

            if (convertLayerTypeToString(predecessors[0]->getType()).compare("aten::transpose") == 0) {
                auto addmm_layer = std::static_pointer_cast<nn_compiler::ir::AtenAddmmLayer>(successors[0]);
                addmm_layer->set_act_type(convertLayerTypeToString(cur_layer->getType()));
            } else {
                predecessors[0]->setActivation(cur_layer);
            }

            predecessors[0]->renewOutSTensorID(0, out_ids[0]);

            layers_to_be_removed.push_back(cur_layer);

            model->updateLayerRelationShips(out_ids[0], cur_layer, predecessors[0]);
        }
    }

    for (auto layer_to_be_removed : layers_to_be_removed) {
        graph->deleteLayer(layer_to_be_removed->getID());
    }
}

bool FuseActivation::feasibleHostType(const std::string& type)
{
    return (std::find(supportd_host_types_.begin(), supportd_host_types_.end(), type) != supportd_host_types_.end());
}

bool FuseActivation::feasibleParasiteType(const std::string& type)
{
    return (std::find(supported_parasite_types_.begin(), supported_parasite_types_.end(), type) !=
            supported_parasite_types_.end());
}

}  // namespace frontend
}  // namespace nn_compiler
