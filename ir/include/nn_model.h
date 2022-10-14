/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "ir/include/nn_graph.h"

namespace nn_compiler
{
namespace ir
{
class NNModel
{
   public:
    NNModel() {}

    void appendGraph(const std::shared_ptr<NNGraph> graph) { graphs_.push_back(graph); }

    void removeGraphs() { graphs_.clear(); }

    void removeGraph(const std::string& graph_name)
    {
        for (auto iter = graphs_.begin(); iter != graphs_.end();) {
            if ((*iter)->getName() == graph_name) {
                iter = graphs_.erase(iter);
            } else {
                iter++;
            }
        }
    }

    void reverseGraphs() { std::reverse(graphs_.begin(), graphs_.end()); }

    std::vector<std::shared_ptr<NNGraph>> getGraphs() { return graphs_; }

    void addSTensor(std::pair<uint32_t, std::shared_ptr<STensor>> shape_tensor)
    {
        shape_tensors_.insert(shape_tensor);
    }

    void deleteSTensor(uint32_t shape_tensor_id) { shape_tensors_.erase(shape_tensor_id); }

    std::map<uint32_t, std::shared_ptr<STensor>> getSTensors() { return shape_tensors_; }

    void addLayerRelationShips(uint32_t shape_tensor_id, std::shared_ptr<ir::NNLayer> layer)
    {
        auto res = layer_relations_map_[shape_tensor_id];
        bool jud = false;
        for (auto layer_ : res) {
            if (layer_ == layer) {
                jud = true;
                break;
            }
        }
        if (!jud) {
            res.push_back(layer);
        }
        layer_relations_map_[shape_tensor_id] = res;
    }

    void updateLayerRelationShips(uint32_t shape_tensor_id, std::shared_ptr<ir::NNLayer> from_layer,
                                  std::shared_ptr<ir::NNLayer> to_layer)
    {
        auto res = layer_relations_map_[shape_tensor_id];
        for (int idx = 0; idx < res.size(); idx++) {
            if (res[idx] == from_layer) {
                res[idx] = to_layer;
            }
        }
        layer_relations_map_[shape_tensor_id] = res;
    }

    void deleteLayerRelationShips(uint32_t shape_tensor_id, std::shared_ptr<ir::NNLayer> layer)
    {
        auto res = layer_relations_map_[shape_tensor_id];
        std::vector<std::shared_ptr<ir::NNLayer>> tmp_res;
        for (auto layer_ : res) {
            if (layer_ != layer) {
                tmp_res.push_back(layer_);
            }
        }
        layer_relations_map_[shape_tensor_id] = tmp_res;
    }

    std::map<uint32_t, std::vector<std::shared_ptr<ir::NNLayer>>> getLayerRelationShips()
    {
        return layer_relations_map_;
    }

   private:
    std::vector<std::shared_ptr<NNGraph>> graphs_;

    std::map<uint32_t, std::shared_ptr<STensor>> shape_tensors_;

    std::map<uint32_t, std::vector<std::shared_ptr<ir::NNLayer>>> layer_relations_map_;
};

}  // namespace ir
}  // namespace nn_compiler
