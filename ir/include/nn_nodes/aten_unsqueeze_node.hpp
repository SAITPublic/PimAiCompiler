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

class AtenUnsqueezeNode : public NodeMixin<AtenUnsqueezeNode, NNNode> {
 public:
    explicit AtenUnsqueezeNode(const NodeInfo& node_info, int64_t dim, bool is_inplace)
        : NodeMixin(node_info, NodeType::ATENUNSQUEEZE), dim_(dim), is_inplace_(is_inplace) {}

    std::string getNodeTypeAsString() const override { return "AtenUnsqueeze"; }

    int64_t getDim() const { return dim_; }

    bool getIsInplace() const { return is_inplace_; }

 private:
    int64_t dim_;
    bool is_inplace_ = false;

}; // class AtenUnsqueezeNode

} // namespace nn_ir
} // namespace nn_compiler
