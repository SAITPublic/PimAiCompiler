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

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class TileNode : public NodeMixin<TileNode, NNNode> {
 public:
    explicit TileNode(const NodeInfo& node_info, nn_ir::Axis axis, uint32_t tiles)
        : NodeMixin(node_info, NodeType::TILE), axis_(axis), tiles_(tiles) {}

    std::string getNodeTypeAsString(void) const override { return "Tile"; }

    nn_ir::Axis getAxis() const { return axis_; }
    uint32_t    getTiles() const { return tiles_; }

 private:
    nn_ir::Axis axis_;
    uint32_t    tiles_;
}; // class TileNode

} // namespace nn_ir
} // namespace nn_compiler
