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

class AtenAddNode : public NodeMixin<AtenAddNode, NNNode> {
 public:
    explicit AtenAddNode(const NodeInfo& node_info, int64_t alpha) :
            NodeMixin(node_info, NodeType::ATENADD), alpha_(alpha) {}

    std::string getNodeTypeAsString(void) const override { return "AtenAdd"; }

    void setAlpha(int64_t alpha) { alpha_ = alpha; }

    int64_t getAlpha() { return alpha_; }

 private:
    int64_t alpha_ = 1;
}; // class AtenAddNode

} // namespace nn_ir
} // namespace nn_compiler
