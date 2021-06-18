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
 * @file.    concat_node.hpp
 * @brief.   This is ConcatNode class
 * @details. This header defines ConcatNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class ConcatNode : public NodeMixin<ConcatNode, NNNode> {
 public:
    explicit ConcatNode(const NodeInfo& node_info, nn_ir::Axis axis)
        : NodeMixin(node_info, NodeType::CONCAT), axis_(axis) {}

    std::string getNodeTypeAsString() const override { return "Concat"; }

    nn_ir::Axis getAxis() const { return axis_; }

    bool isReshaping() const;

    // TODO(a.puschin): simplify this after CRANC-680
    nn_ir::Coord4D getConcatenationOffsetFor(const nn_ir::Node& predecessor_node) const;

 private:
    // TODO(a.puschin): add a map of 4D offsets for input blobs here (see CRANC-680 for detail)
    nn_ir::Axis axis_;
}; // class ConcatNode

} // namespace nn_ir
} // namespace nn_compiler
