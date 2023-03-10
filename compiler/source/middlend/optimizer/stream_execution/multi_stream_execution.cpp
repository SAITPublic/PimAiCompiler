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
            std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> multi_stream_apply_layers;
            if (!backwardToCheckAndFindStartLayer(model, multi_stream_start_layer, multi_stream_apply_layers)) {
                branches_.clear();
                continue;
            }
            start_to_stream_layers_[multi_stream_start_layer] = multi_stream_apply_layers;

            branches_.clear();
        }
    }

    return (start_to_stream_layers_.size() != 0);
}

void MultiStreamExecution::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "MultiStreamExecution::run is called.";
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();
    auto layer_size = layers.size();

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> start_layers;
    for (auto pairs : start_to_stream_layers_) {
        start_layers.push_back(pairs.first);
    }

    for (auto idx = 0; idx < layer_size; idx++) {
        if (std::find(start_layers.begin(), start_layers.end(), layers[idx]) != start_layers.end()) {
            // reorganize layer oders and create multi-stream layer
            process(model, idx);

            layers = graph->getLayers();
            layer_size = layers.size();
        }
    }
}

bool MultiStreamExecution::backwardToCheckAndFindStartLayer(
    std::unique_ptr<nn_compiler::ir::NNModel>& model,
    std::shared_ptr<nn_compiler::ir::NNLayer>& multi_stream_start_layer,
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>& multi_stream_apply_layers)
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

    for (auto i = 0; i < branches_.size(); i++) {
        multi_stream_apply_layers.push_back(branches_[i][stream_op_index_in_branches[i]]);
    }

    return (multi_stream_start_layer != nullptr && multi_stream_apply_layers.size() != 0);
}

void MultiStreamExecution::process(std::unique_ptr<nn_compiler::ir::NNModel>& model, int start_idx)
{
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();
    auto start_layer = layers[start_idx];

    auto multi_stream_apply_layers = start_to_stream_layers_[start_layer];
    int found_num = 0;
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> pre_process_layers;
    std::queue<std::shared_ptr<nn_compiler::ir::NNLayer>> processor;
    for (auto layer : multi_stream_apply_layers) {
        processor.push(layer);
    }

    while (!processor.empty() && found_num < multi_stream_apply_layers.size()) {
        auto cur_layer = processor.front();
        processor.pop();
        auto predecessors = pre_layers_map_[cur_layer];
        for (auto predecessor : predecessors) {
            if (predecessor->getType() == ir::LayerType::PRIMCONSTANT) {
                continue;
            }
            if (predecessor == start_layer) {
                found_num++;
            } else {
                pre_process_layers.push_back(predecessor);
                processor.push(predecessor);
            }
        }
    }
    std::reverse(pre_process_layers.begin(), pre_process_layers.end());

    int new_idx = start_idx;
    for (auto layer : pre_process_layers) {
        graph->deleteLayer(layer);
        graph->addLayer2pos(layer, new_idx++);
    }
    for (auto layer : multi_stream_apply_layers) {
        graph->deleteLayer(layer);
    }

    auto new_type = nn_compiler::ir::LayerType::MULTISTREAM;
    auto new_name = "multi_stream_for_" + multi_stream_apply_layers[0]->getName();
    auto new_layer = std::make_shared<nn_compiler::ir::MultiStreamLayer>(new_name, new_type);
    new_layer->setLayers(multi_stream_apply_layers);
    new_layer->setLayerNum(multi_stream_apply_layers.size());
    graph->addLayer2pos(new_layer, new_idx++);

    // add back multi_stream_apply_layers to guarantee contiguous layer IDs for the following UpdateLayerId pass
    for (auto layer : multi_stream_apply_layers) {
        graph->addLayer2pos(layer, new_idx++);
    }
}

}  // namespace middlend
}  // namespace nn_compiler
