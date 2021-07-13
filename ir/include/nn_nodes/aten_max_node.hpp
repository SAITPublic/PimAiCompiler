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

class AtenMaxNode : public NodeMixin<AtenMaxNode, NNNode> {
 public:
    explicit AtenMaxNode(const NodeInfo& node_info, int64_t dim, bool keep_dim)
            : NodeMixin(node_info, NodeType::ATENMAX), dim_(dim), keep_dim_(keep_dim) {}

    std::string getNodeTypeAsString(void) const override { return "AtenMax"; }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() { return dim_; }

    void setKeepDim(bool keep_dim) { keep_dim_ = keep_dim; }

    bool getKeepDim() { return keep_dim_; }

private:
    int64_t dim_   = 0;
    bool keep_dim_ = false;
}; // class AtenMaxNode

} // namespace nn_ir
} // namespace nn_compiler
