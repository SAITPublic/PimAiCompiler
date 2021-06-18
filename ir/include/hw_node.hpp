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

#include "ir/include/ir_types.hpp"
#include "ir/include/node.hpp"

#include "common/include/common.hpp"
#include "common/include/types.hpp"

namespace nn_compiler::nn_ir {

class HWNode : public AbstractNodeMixin<HWNode, Node> {
 public:
    explicit HWNode(const NodeInfo& node_info) : HWNode(node_info, NodeType::HWNode) {}

    /// @brief methods used by casting infrastructure
    template <typename T>
    static bool classof(const Node* node) {
        static_assert(std::is_same<T, HWNode>::value, "incorrect type");
        return node->getNodeType() >= NodeType::HWNode && node->getNodeType() <= NodeType::LastHWNode;
    }

 protected:
    HWNode(const HWNode&) = default;
    HWNode(HWNode&&)      = default;

    // constructor for inherited classes
    explicit HWNode(const NodeInfo& node_info, NodeType node_type) : AbstractNodeMixin(node_info, node_type) {}

    explicit HWNode(const Node& node) : AbstractNodeMixin(node) {}
}; // class HWNode

} // namespace nn_compiler::nn_ir
