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
 * @file.    op_node.hpp
 * @brief.   This is OPNode class
 * @details. This header defines OPNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/node.hpp"

namespace nn_compiler {
namespace nn_ir {

class OPNode : public AbstractNodeMixin<OPNode, Node> {
 public:
    explicit OPNode(const NodeInfo& node_info) : OPNode(node_info, NodeType::OPNode) {}

 protected:
    OPNode(const OPNode&) = default;
    OPNode(OPNode&&)      = default;

    // constructor for inherited classes
    explicit OPNode(const NodeInfo& node_info, NodeType node_type) : AbstractNodeMixin(node_info, node_type) {}
}; // class OPNode

} // namespace nn_ir
} // namespace nn_compiler
