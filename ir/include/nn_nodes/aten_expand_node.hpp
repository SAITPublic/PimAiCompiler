/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
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

class AtenExpandNode : public NodeMixin<AtenExpandNode, NNNode> {
 public:
    explicit AtenExpandNode(const NodeInfo& node_info, int implicit):
            NodeMixin(node_info, NodeType::ATENEXPAND), implicit_(implicit) {}

    std::string getNodeTypeAsString(void) const override { return "AtenExpand"; }

    void setImplicit(int implicit) { implicit_ = implicit; }

    int getImplicit() { return implicit_; }

private:
    int implicit_;
}; // class AtenExpandNode

} // namespace nn_ir
} // namespace nn_compiler
