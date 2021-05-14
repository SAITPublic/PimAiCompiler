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
#include "ir/node.hpp"

namespace nn_compiler {
namespace nn_ir {

class QNode : public AbstractNodeMixin<QNode, Node> {
 public:
    explicit QNode(const NodeInfo& node_info) : QNode(node_info, NodeType::QNode) {}

 protected:
    QNode(const QNode&) = default;
    QNode(QNode&&)      = default;

    // constructor for inherited classes
    explicit QNode(const NodeInfo& node_info, NodeType node_type) : AbstractNodeMixin(node_info, node_type) {}
}; // class QNode

} // namespace nn_ir
} // namespace nn_compiler
