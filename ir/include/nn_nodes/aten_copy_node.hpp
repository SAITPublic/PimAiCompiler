/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenCopyNode : public NodeMixin<AtenCopyNode, NNNode> {
 public:
    explicit AtenCopyNode(const NodeInfo& node_info, bool non_blocking)
        : NodeMixin(node_info, NodeType::ATENCOPY), non_blocking_(non_blocking) {}

    std::string getNodeTypeAsString() const override { return "AtenCopy"; }

    bool getNonBlocking() const { return non_blocking_; }

 private:
    bool non_blocking_;
}; // class AtenCopyNode

} // namespace nn_ir
} // namespace nn_compiler
