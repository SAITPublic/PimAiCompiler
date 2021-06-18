/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    dummy_node.hpp
 * @brief.   This is DummyNode class
 * @details. This header defines DummyNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {
class DummyNode : public NNNode {
 public:
    explicit DummyNode(const NodeInfo& node_info) : NNNode(node_info, NodeType::DUMMY) {}

    std::unique_ptr<DummyNode> clone() const& { return std::unique_ptr<DummyNode>(this->cloneImpl()); }
    std::unique_ptr<DummyNode> clone() && { return std::unique_ptr<DummyNode>(std::move(*this).cloneImpl()); }

    std::string getNodeTypeAsString() const override { return "Dummy"; }

    template <typename T>
    static bool classof(const Node* node) {
        static_assert(std::is_same<T, DummyNode>::value, "incorrect type");
        return node->getNodeType() == NodeType::DUMMY;
    }

 private:
    DummyNode(const DummyNode&) = default;
    DummyNode(DummyNode&&)      = default;

    DummyNode* cloneImpl() const& override { return new DummyNode(*this); }
    DummyNode* cloneImpl() && override { return new DummyNode(std::move(*this)); }
}; // class DummyNode
} // namespace nn_ir
} // namespace nn_compiler
