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

class AtenClampNode : public NodeMixin<AtenClampNode, NNNode> {
 public:
    explicit AtenClampNode(const NodeInfo& node_info, int min, int max) :
            NodeMixin(node_info, NodeType::ATENCLAMP), min_(min), max_(max) {}

    std::string getNodeTypeAsString(void) const override { return "AtenClamp"; }

    void setMin(int min) { min_ = min; }

    int getMin() { return min_; }

    void setMax(int max) { max_ = max; }

    int getMax() { return max_; }

private:
    int min_ = INT32_MAX;
    int max_ = INT32_MAX;

}; // class AtenClampNode

} // namespace nn_ir
} // namespace nn_compiler
