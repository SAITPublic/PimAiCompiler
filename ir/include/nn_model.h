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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ir/include/nn_network.h"
#include "ir/include/tensors/torch_shape_tensor.h"

namespace nn_compiler {
namespace ir {

class NNModel {
 public:
    NNModel() {
    }

    void appendGraph(const std::shared_ptr<NNNetwork> graph) {
        graphs_.push_back(graph);
    }

    void removeGraphs() {
        graphs_.clear();
    }

    void removeGraph(const std::string& graph_name) {
        for (auto iter = graphs_.begin(); iter != graphs_.end();) {
            if ((*iter)->getName() == graph_name) {
                iter = graphs_.erase(iter);
            } else {
                iter++;
            }
        }
    }

    void reverseGraphs() {
        std::reverse(graphs_.begin(), graphs_.end());
    }

    std::vector<std::shared_ptr<NNNetwork>> getGraphs() {
        return graphs_;
    }

    void addTSSTensor(std::pair<uint32_t, std::shared_ptr<TSSTensor>> shape_tensor) {
        shape_tensors_.insert(shape_tensor);
    }

    void deleteTSSTensor(uint32_t shape_tensor_id) {
        shape_tensors_.erase(shape_tensor_id);
    }

    std::map<uint32_t, std::shared_ptr<TSSTensor>> getTSSTensors() {
        return shape_tensors_;
    }

 private:
    std::vector<std::shared_ptr<NNNetwork>> graphs_;

    std::map<uint32_t, std::shared_ptr<TSSTensor>> shape_tensors_;
};

}  // namespace ir
}  // namespace nn_compiler
