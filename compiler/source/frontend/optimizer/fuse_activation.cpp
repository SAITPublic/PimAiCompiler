#include "compiler/include/frontend/optimizer/fuse_activation.h"
#include "new_ir/include/layers/aten_addmm_layer.h"

#include "new_ir/include/utils/graph_search.h"
#include "new_ir/include/utils/graph_transform.h"
#include "new_ir/include/utils/graph_util.h"

#include "compiler/include/common/log.hpp"

namespace nn_compiler {

namespace frontend {

FuseActivation::FuseActivation() {
}

bool FuseActivation::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    auto dependCheck = [&] (const std::shared_ptr<nn_compiler::ir::NNNetwork> network,
                            const std::shared_ptr<nn_compiler::ir::NNLayer> predecessor,
                            const std::shared_ptr<nn_compiler::ir::NNLayer> successor) {
        std::string predecessor_type = convertLayerTypeToString(predecessor->getType());
        if (!this->feasibleHostType(predecessor_type)) {
            Log::FE::D() << "failed to satisfy with fusion dependency";
            return false;
        }

        auto successors = ir::searchSuccessor(predecessor, network);
        if (successors.size() > 1) {
            Log::FE::D() << "failed to satisfy with fusion dependency";
            return false;
        }

        if (predecessor_type.compare("aten::transpose") == 0) {
            if (successor->getType() != nn_compiler::ir::LayerType::ATENADDMM) {
                Log::FE::D() << "The predecessor of transpose layer is not addmm layer.";
                return false;
            }
            auto successors = ir::searchSuccessor(successor, network);
            if (successors.size() > 1) {
                Log::FE::D() << "failed to satisfy with fusion dependency";
                return false;
            }
        }
        return true;
    };

    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto cur_layer : graph->getLayers()) {
            std::string type = convertLayerTypeToString(cur_layer->getType());
            if (feasibleParasiteType(type)) {
                auto predecessors = ir::searchPredecessor(cur_layer, graph);
                auto successors = ir::searchPredecessor(predecessors[0], graph);

                if (predecessors.empty() || !dependCheck(graph, predecessors[0], successors[0])) {
                    continue;
                }

                layers_.push_back(cur_layer);
            }
        }
    }

    return (layers_.size() != 0);
}

void FuseActivation::run(std::unique_ptr<nn_compiler::ir::NNModel>& model) {
    Log::FE::I() << "FuseActivation::run is called.";
    auto graph = model->getGraphs()[0];

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_to_be_removed;
    std::vector<uint32_t > tensors_to_be_removed;
    for (auto cur_layer : layers_) {
        std::string type = convertLayerTypeToString(cur_layer->getType());
        if (feasibleParasiteType(type)) {
            auto predecessors = ir::searchPredecessor(cur_layer, graph);
            auto successors = ir::searchPredecessor(predecessors[0], graph);

            auto out_ids = cur_layer->getOutSTensorID();
            auto in_ids = cur_layer->getInSTensorID();
            CHECK_EQ(out_ids.size(), 1);
            CHECK_EQ(in_ids.size(), 1);
            tensors_to_be_removed.push_back(in_ids[0]);

            if (convertLayerTypeToString(predecessors[0]->getType()).compare("aten::transpose") == 0) {
                auto addmm_layer = std::static_pointer_cast<nn_compiler::ir::AtenAddmmLayer>(successors[0]);
                addmm_layer->set_act_type(convertLayerTypeToString(cur_layer->getType()));
            } else {
                predecessors[0]->setActivation(cur_layer);
            }

            predecessors[0]->renewOutSTensorID(0, out_ids[0]);

            layers_to_be_removed.push_back(cur_layer);
        }
    }

    for (auto layer_to_be_removed : layers_to_be_removed) {
        graph->deleteLayer(layer_to_be_removed->getID());
    }

    for (auto tensor_to_be_removed : tensors_to_be_removed) {
        graph->deleteSTensor(tensor_to_be_removed);
    }
}

bool FuseActivation::feasibleHostType(const std::string &type) {
    return (std::find(supportd_host_types_.begin(),
                      supportd_host_types_.end(),
                      type) != supportd_host_types_.end());
}

bool FuseActivation::feasibleParasiteType(const std::string &type) {
    return (std::find(supported_parasite_types_.begin(),
                      supported_parasite_types_.end(),
                      type) != supported_parasite_types_.end());
}

}  // namespace frontend
}  // namespace nn_compiler
