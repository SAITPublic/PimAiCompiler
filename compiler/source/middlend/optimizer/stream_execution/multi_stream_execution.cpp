#include <string>

#include "ir/include/utils/graph_util.h"
#include "middlend/optimizer/stream_execution/multi_stream_execution.h"

namespace nn_compiler
{
namespace middlend
{
MultiStreamExecution::MultiStreamExecution() {}

bool MultiStreamExecution::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();

    for (auto idx = 0; idx < layers.size(); idx++) {
        auto predecessors = ir::utils::searchPredecessor(layers[idx], model);
        pre_layers_map_[layers[idx]] = predecessors;
    }

    for (auto idx = 0; idx < layers.size(); idx++) {
        if (layers[idx]->getType() == ir::LayerType::PRIMENDIF) {
            // if&else branches shouldn't be parrallel
            continue;
        }

        auto predecessors = pre_layers_map_[layers[idx]];
        int non_constant_layer_num = 0;
        for (auto predecessor : predecessors) {
            if (predecessor->getType() != ir::LayerType::PRIMCONSTANT) {
                non_constant_layer_num++;
            }
        }
        if (non_constant_layer_num > 1) {
            for (auto predecessor : predecessors) {
                if (predecessor->getType() != ir::LayerType::PRIMCONSTANT) {
                    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> new_branch;
                    new_branch.push_back(predecessor);
                    branches_.push_back(new_branch);
                }
            }

            std::shared_ptr<nn_compiler::ir::NNLayer> multi_stream_start_layer = nullptr;
            if (!backwardToCheckAndFindStartLayer(model, multi_stream_start_layer)) {
                branches_.clear();
                continue;
            }
            if (multi_stream_start_layer != nullptr) {
                multi_stream_layers_.push_back(multi_stream_start_layer);
                multi_stream_layers_.push_back(layers[idx]);
            }

            branches_.clear();
        }
    }

    return (multi_stream_layers_.size() != 0);
}

void MultiStreamExecution::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "MultiStreamExecution::run is called.";
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();

    for (int idx = 0; idx < layers.size(); idx++) {
        auto layer = layers[idx];
        if (std::find(multi_stream_layers_.begin(), multi_stream_layers_.end(), layer) != multi_stream_layers_.end()) {
            idx = reorganizeLayerOrders(graph, idx);
        }
    }

    layers = graph->getLayers();
    for (int idx = 0; idx < layers.size(); idx++) {
        auto layer = layers[idx];
        if (std::find(multi_stream_layers_.begin(), multi_stream_layers_.end(), layer) != multi_stream_layers_.end() &&
            layer->getType() != ir::LayerType::PRIMLISTCONSTRUCT) {
            insertMultiStreamLayer(graph, idx);
            layers = graph->getLayers();
        }
    }
}

bool MultiStreamExecution::backwardToCheckAndFindStartLayer(
    std::unique_ptr<nn_compiler::ir::NNModel>& model,
    std::shared_ptr<nn_compiler::ir::NNLayer>& multi_stream_start_layer)
{
    std::vector<int> search_index_of_branches(branches_.size(), 0);
    // backward to find branch ops within max_search_level_
    for (auto i = 0; i < branches_.size(); i++) {
        auto& branch = branches_[i];
        bool found_stream_support_op = false;
        for (int cnt = 0; cnt < max_search_level_; cnt++) {
            auto origin_branch_size = branch.size();
            for (auto& index = search_index_of_branches[i]; index < origin_branch_size; index++) {
                if (branch[index]->getType() == ir::LayerType::PRIMENDIF) {
                    break;
                }
                if (branch[index]->getType() == ir::LayerType::ATENADD ||
                    branch[index]->getType() == ir::LayerType::ATENADDMM ||
                    branch[index]->getType() == ir::LayerType::ATENMATMUL) {
                    // currently, only add, addmm and matmul support stream argument.
                    // other Ops are calling aten lib directly.
                    found_stream_support_op = true;
                }

                auto predecessors = pre_layers_map_[branch[index]];
                for (auto predecessor : predecessors) {
                    if (predecessor->getType() != ir::LayerType::PRIMCONSTANT) {
                        branch.push_back(predecessor);
                    }
                }
            }
        }
        if (!found_stream_support_op) {
            return false;
        }
    }

    std::vector<int> stream_op_index_in_branches(branches_.size(), -1);
    for (auto i = 0; i < branches_.size(); i++) {
        auto& branch = branches_[i];
        for (auto j = 0; j < branch.size(); j++) {
            if (branch[j]->getType() == ir::LayerType::ATENMATMUL || branch[j]->getType() == ir::LayerType::ATENADDMM) {
                // TODO: for current supported model, addmm and matmul spend much more time than add.
                // It may be possible for us to support mult-stream for add later, when large add is found in the model.
                stream_op_index_in_branches[i] = j;
                break;
            }
        }
    }
    for (auto i = 0; i < stream_op_index_in_branches.size(); i++) {
        if (stream_op_index_in_branches[i] == -1) {  // double-check no stream-support ops for parralelism
            return false;
        }
    }

    // find start layer of parralelism module
    std::set<std::pair<int, std::shared_ptr<nn_compiler::ir::NNLayer>>> candidate_starts_set;  // pair: (level, layer)
    for (auto i = 0; i < branches_.size(); i++) {
        auto& branch = branches_[i];
        if (candidate_starts_set.size() == 0) {
            if (i == 0) {
                for (auto j = stream_op_index_in_branches[i]; j < branch.size(); j++) {
                    candidate_starts_set.insert(std::make_pair((j - stream_op_index_in_branches[i] + 1), branch[j]));
                }
            } else {
                return false;
            }
        }
        std::set<std::pair<int, std::shared_ptr<nn_compiler::ir::NNLayer>>> updated_set;
        for (auto j = stream_op_index_in_branches[i]; j < branch.size(); j++) {
            auto item = std::make_pair((j - stream_op_index_in_branches[i] + 1), branch[j]);
            if (candidate_starts_set.find(item) != candidate_starts_set.end()) {
                updated_set.insert(item);
            }
        }
        candidate_starts_set = updated_set;
    }
    if (candidate_starts_set.size() == 0 ||
        candidate_starts_set.begin()->second == branches_[0][stream_op_index_in_branches[0]]) {
        // no same start, or the start layer is just the parralelism layer
        return false;
    }

    // the module may be nested case (more than one layer has multiple input branches).
    // check whether the candidate start is just the predecessor of stream ops
    for (auto candidate_start : candidate_starts_set) {
        std::queue<std::shared_ptr<nn_compiler::ir::NNLayer>> stream_branch_ops;
        bool valid_start = true;
        for (auto i = 0; i < branches_.size(); i++) {
            stream_branch_ops.push(branches_[i][stream_op_index_in_branches[i]]);
            bool found_in_branch = false;
            int cnt = 0;
            while (!stream_branch_ops.empty() && cnt++ < max_search_level_) {
                auto cur_layer = stream_branch_ops.front();
                stream_branch_ops.pop();
                auto pre_layers = pre_layers_map_[cur_layer];
                for (auto pre_layer : pre_layers) {
                    if (pre_layer == candidate_start.second) {
                        found_in_branch = true;
                        break;
                    }
                    stream_branch_ops.push(pre_layer);
                }
                if (found_in_branch) {
                    break;
                }
            }
            if (!found_in_branch) {
                valid_start = false;
                break;
            }
        }
        if (valid_start) {
            multi_stream_start_layer = candidate_start.second;
            break;
        }
    }

    return (multi_stream_start_layer != nullptr);
}

int MultiStreamExecution::reorganizeLayerOrders(std::shared_ptr<ir::NNGraph>& graph, int start_idx)
{
    auto layers = graph->getLayers();
    int idx = start_idx + 1;

    std::vector<ir::LayerType> layer_types;
    layer_types.push_back(layers[idx]->getType());
    assert((idx + 1) < layers.size());
    // Presupposition: no same datatype for Ops in each stream. Remove this condition later.
    for (int i = idx + 1; i < layers.size(); i++) {
        if (layers[i]->getType() == layer_types[0]) {
            break;
        } else {
            layer_types.push_back(layers[i]->getType());
        }
    }

    int end_idx = start_idx + 1;
    for (end_idx; end_idx < layers.size(); end_idx++) {
        if (std::find(multi_stream_layers_.begin(), multi_stream_layers_.end(), layers[end_idx]) !=
            multi_stream_layers_.end()) {
            break;
        }
    }

    int swap_idx = idx + 1;
    for (auto layer_type : layer_types) {
        layers = graph->getLayers();
        for (int i = swap_idx; i < end_idx; i++) {
            if (layers[i]->getType() == layer_type) {
                graph->swapLayerOrder(swap_idx++, i);
            }
        }
    }

    return end_idx;
}

void MultiStreamExecution::insertMultiStreamLayer(std::shared_ptr<ir::NNGraph>& graph, int idx)
{
    auto layers = graph->getLayers();
    std::vector<std::shared_ptr<ir::NNLayer>> matmul_layers;
    bool find_matmul = false;
    for (idx; idx < layers.size(); idx++) {
        while (layers[idx]->getType() == ir::LayerType::ATENMATMUL) {
            matmul_layers.push_back(layers[idx++]);
            find_matmul = true;
        }
        if (find_matmul) {
            break;
        }
    }
    auto new_type = nn_compiler::ir::LayerType::MULTISTREAM;
    auto new_name = "multi_stream_for_" + matmul_layers[0]->getName();
    auto new_layer = std::make_shared<nn_compiler::ir::MultiStreamLayer>(new_name, new_type);
    new_layer->setLayers(matmul_layers);
    new_layer->setLayerNum(matmul_layers.size());
    new_layer->setStreams();
    graph->addLayer2pos(new_layer, idx - matmul_layers.size() - 1);
}

}  // namespace middlend
}  // namespace nn_compiler
