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
 * @file.    slice_node.hpp
 * @brief.   This is SliceNode class
 * @details. This header defines SliceNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class SliceNode : public NodeMixin<SliceNode, NNNode> {
 public:
    SliceNode(const NodeInfo& node_info, nn_ir::Axis axis, std::vector<uint8_t> points)
        : NodeMixin(node_info, NodeType::SLICE), axis_(axis), points_(points) {}

    std::string getNodeTypeAsString() const override { return "Slice"; }

    void setAxis(nn_ir::Axis axis) { axis_ = axis; }
    void setPoints(std::vector<uint8_t> points) { points_ = points; }

    nn_ir::Axis          getAxis() const { return axis_; }
    std::vector<uint8_t> getPoints() const { return points_; }

 private:
    nn_ir::Axis          axis_;
    std::vector<uint8_t> points_;
}; // class SliceNode

} // namespace nn_ir
} // namespace nn_compiler
