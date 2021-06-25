/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/node.hpp"

namespace nn_compiler {
namespace nn_ir {

class NNNode : public AbstractNodeMixin<NNNode, Node> {
 public:
    explicit NNNode(const NodeInfo& node_info) : NNNode(node_info, NodeType::NNNode) {}

    void setTargetDevice(const std::string& target_device) { target_device_ = target_device; }

    const std::string& getTargetDevice() { return target_device_; }

 protected:
    NNNode(const NNNode&) = default;
    NNNode(NNNode&&)      = default;

    // constructor for inherited classes
    explicit NNNode(const NodeInfo& node_info, NodeType node_type) : AbstractNodeMixin(node_info, node_type) {}

 private:
    std::string target_device_ = "CPU";
}; // class NNNode

} // namespace nn_ir
} // namespace nn_compiler
