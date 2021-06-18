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
 * @file.    permute_node.hpp
 * @brief.   This is PermuteNode class
 * @details. This header defines PermuteNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PermuteNode : public NodeMixin<PermuteNode, NNNode> {
 public:
    explicit PermuteNode(const NodeInfo& node_info,
                         Shape4D         permute_order,
                         Shape4D         input_shape = {{.n = 1, .c = 1, .h = 1, .w = 1}})
        : NodeMixin(node_info, NodeType::PERMUTE), permute_order_(permute_order), in_shape_(input_shape) {}

    std::string getNodeTypeAsString() const override { return "Permute"; }

    const Shape4D& getPermuteOrder() const { return permute_order_; }
    const Shape4D& getInputShape() const { return in_shape_; }
    bool           isUsingCellInput() const { return use_cell_input_; }
    void           setUsingCellInput(bool v) { use_cell_input_ = v; }

 private:
    Shape4D permute_order_;
    Shape4D in_shape_;
    bool    use_cell_input_ = false;
}; // class PermuteNode

} // namespace nn_ir
} // namespace nn_compiler
