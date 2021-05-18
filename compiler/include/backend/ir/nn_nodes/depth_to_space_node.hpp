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
 * @file.    depth_to_space_node.hpp
 * @brief.   This is DepthToSpaceNode class
 * @details. This header defines DepthToSpaceNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

#include "ir/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class DepthToSpaceNode : public NodeMixin<DepthToSpaceNode, NNNode> {
 public:
    explicit DepthToSpaceNode(const NodeInfo& node_info, int block_size)
        : NodeMixin(node_info, NodeType::DEPTHTOSPACE), block_size_(block_size) {}

    std::string getNodeTypeAsString() const override { return "DepthToSpace"; }

    int getBlockSize() const { return block_size_; }

 private:
    int block_size_;
}; // class DepthToSpaceNode

} // namespace nn_ir
} // namespace nn_compiler