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
 * @file.    reshape_node.hpp
 * @brief.   This is ReshapeNode class
 * @details. This header defines ReshapeNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class ReshapeNode : public NodeMixin<ReshapeNode, NNNode> {
 public:
    explicit ReshapeNode(const NodeInfo& node_info, Shape4D shape)
        : NodeMixin(node_info, NodeType::RESHAPE), shape_(shape) {}

    std::string getNodeTypeAsString() const override { return "Reshape"; }

    const Shape4D& getShape() const { return shape_; }

 private:
    Shape4D shape_;
}; // class ReshapeNode

} // namespace nn_ir
} // namespace nn_compiler
